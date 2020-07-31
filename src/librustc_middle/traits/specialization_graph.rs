use crate::ich::{self, StableHashingContext};
use crate::ty::fast_reject::SimplifiedType;
use crate::ty::fold::TypeFoldable;
use crate::ty::{self, TyCtxt};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_errors::ErrorReported;
use rustc_hir::def_id::{DefId, DefIdMap};
use rustc_span::symbol::Ident;

/// A per-trait graph of impls in specialization order. At the moment, this
/// graph forms a tree rooted with the trait itself, with all other nodes
/// representing impls, and parent-child relationships representing
/// specializations.
///
/// The graph provides two key services:
///
/// - Construction. This implicitly checks for overlapping impls (i.e., impls
///   that overlap but where neither specializes the other -- an artifact of the
///   simple "chain" rule.
///
/// - Parent extraction. In particular, the graph can give you the *immediate*
///   parents of a given specializing impl, which is needed for extracting
///   default items amongst other things. In the simple "chain" rule, every impl
///   has at most one parent.
#[derive(RustcEncodable, RustcDecodable, HashStable)]
pub struct Graph {
    /// All impls have a parent; the "root" impls have as their parent the `def_id`
    /// of the trait.
    pub parent: DefIdMap<DefId>,

    /// The "root" impls are found by looking up the trait's def_id.
    pub children: DefIdMap<Children>,

    /// Whether an error was emitted while constructing the graph.
    pub has_errored: bool,
}

impl Graph {
    pub fn new() -> Graph {
        Graph { parent: Default::default(), children: Default::default(), has_errored: false }
    }

    /// The parent of a given impl, which is the `DefId` of the trait when the
    /// impl is a "specialization root".
    pub fn parent(&self, child: DefId) -> DefId {
        *self.parent.get(&child).unwrap_or_else(|| panic!("Failed to get parent for {:?}", child))
    }
}

/// Children of a given impl, grouped into blanket/non-blanket varieties as is
/// done in `TraitDef`.
#[derive(Default, RustcEncodable, RustcDecodable)]
pub struct Children {
    // Impls of a trait (or specializations of a given impl). To allow for
    // quicker lookup, the impls are indexed by a simplified version of their
    // `Self` type: impls with a simplifiable `Self` are stored in
    // `nonblanket_impls` keyed by it, while all other impls are stored in
    // `blanket_impls`.
    //
    // A similar division is used within `TraitDef`, but the lists there collect
    // together *all* the impls for a trait, and are populated prior to building
    // the specialization graph.
    /// Impls of the trait.
    pub nonblanket_impls: FxHashMap<SimplifiedType, Vec<DefId>>,

    /// Blanket impls associated with the trait.
    pub blanket_impls: Vec<DefId>,
}

/// A node in the specialization graph is either an impl or a trait
/// definition; either can serve as a source of item definitions.
/// There is always exactly one trait definition node: the root.
#[derive(Debug, Copy, Clone)]
pub enum Node {
    Impl(DefId),
    Trait(DefId),
}

impl<'tcx> Node {
    pub fn is_from_trait(&self) -> bool {
        match *self {
            Node::Trait(..) => true,
            _ => false,
        }
    }

    /// Iterate over the items defined directly by the given (impl or trait) node.
    pub fn items(&self, tcx: TyCtxt<'tcx>) -> impl 'tcx + Iterator<Item = &'tcx ty::AssocItem> {
        tcx.associated_items(self.def_id()).in_definition_order()
    }

    /// Finds an associated item defined in this node.
    ///
    /// If this returns `None`, the item can potentially still be found in
    /// parents of this node.
    pub fn item(
        &self,
        tcx: TyCtxt<'tcx>,
        trait_item_name: Ident,
        trait_item_kind: ty::AssocKind,
        trait_def_id: DefId,
    ) -> Option<ty::AssocItem> {
        tcx.associated_items(self.def_id())
            .filter_by_name_unhygienic(trait_item_name.name)
            .find(move |impl_item| {
                trait_item_kind == impl_item.kind
                    && tcx.hygienic_eq(impl_item.ident, trait_item_name, trait_def_id)
            })
            .copied()
    }

    pub fn def_id(&self) -> DefId {
        match *self {
            Node::Impl(did) => did,
            Node::Trait(did) => did,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Ancestors<'tcx> {
    trait_def_id: DefId,
    specialization_graph: &'tcx Graph,
    current_source: Option<Node>,
}

impl Iterator for Ancestors<'_> {
    type Item = Node;
    fn next(&mut self) -> Option<Node> {
        let cur = self.current_source.take();
        if let Some(Node::Impl(cur_impl)) = cur {
            let parent = self.specialization_graph.parent(cur_impl);

            self.current_source = if parent == self.trait_def_id {
                Some(Node::Trait(parent))
            } else {
                Some(Node::Impl(parent))
            };
        }
        cur
    }
}

/// Information about the most specialized definition of an associated item.
pub struct LeafDef {
    /// The associated item described by this `LeafDef`.
    pub item: ty::AssocItem,

    /// The node in the specialization graph containing the definition of `item`.
    pub defining_node: Node,

    /// The "top-most" (ie. least specialized) specialization graph node that finalized the
    /// definition of `item`.
    ///
    /// Example:
    ///
    /// ```
    /// trait Tr {
    ///     fn assoc(&self);
    /// }
    ///
    /// impl<T> Tr for T {
    ///     default fn assoc(&self) {}
    /// }
    ///
    /// impl Tr for u8 {}
    /// ```
    ///
    /// If we start the leaf definition search at `impl Tr for u8`, that impl will be the
    /// `finalizing_node`, while `defining_node` will be the generic impl.
    ///
    /// If the leaf definition search is started at the generic impl, `finalizing_node` will be
    /// `None`, since the most specialized impl we found still allows overriding the method
    /// (doesn't finalize it).
    pub finalizing_node: Option<Node>,
}

impl LeafDef {
    /// Returns whether this definition is known to not be further specializable.
    pub fn is_final(&self) -> bool {
        self.finalizing_node.is_some()
    }
}

impl<'tcx> Ancestors<'tcx> {
    /// Finds the bottom-most (ie. most specialized) definition of an associated
    /// item.
    pub fn leaf_def(
        mut self,
        tcx: TyCtxt<'tcx>,
        trait_item_name: Ident,
        trait_item_kind: ty::AssocKind,
    ) -> Option<LeafDef> {
        let trait_def_id = self.trait_def_id;
        let mut finalizing_node = None;

        self.find_map(|node| {
            if let Some(item) = node.item(tcx, trait_item_name, trait_item_kind, trait_def_id) {
                if finalizing_node.is_none() {
                    let is_specializable = item.defaultness.is_default()
                        || tcx.impl_defaultness(node.def_id()).is_default();

                    if !is_specializable {
                        finalizing_node = Some(node);
                    }
                }

                Some(LeafDef { item, defining_node: node, finalizing_node })
            } else {
                // Item not mentioned. This "finalizes" any defaulted item provided by an ancestor.
                finalizing_node = Some(node);
                None
            }
        })
    }
}

/// Walk up the specialization ancestors of a given impl, starting with that
/// impl itself.
///
/// Returns `Err` if an error was reported while building the specialization
/// graph.
pub fn ancestors(
    tcx: TyCtxt<'tcx>,
    trait_def_id: DefId,
    start_from_impl: DefId,
) -> Result<Ancestors<'tcx>, ErrorReported> {
    let specialization_graph = tcx.specialization_graph_of(trait_def_id);

    if specialization_graph.has_errored || tcx.type_of(start_from_impl).references_error() {
        Err(ErrorReported)
    } else {
        Ok(Ancestors {
            trait_def_id,
            specialization_graph,
            current_source: Some(Node::Impl(start_from_impl)),
        })
    }
}

impl<'a> HashStable<StableHashingContext<'a>> for Children {
    fn hash_stable(&self, hcx: &mut StableHashingContext<'a>, hasher: &mut StableHasher) {
        let Children { ref nonblanket_impls, ref blanket_impls } = *self;

        ich::hash_stable_trait_impls(hcx, hasher, blanket_impls, nonblanket_impls);
    }
}
