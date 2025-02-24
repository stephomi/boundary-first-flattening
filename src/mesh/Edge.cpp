#include "bff/mesh/Edge.h"
#include "bff/mesh/Mesh.h"

namespace bff {

Edge::Edge(Mesh *mesh_):
onGenerator(false),
onCut(false),
isCuttable(true),
index(-1),
halfEdgeIndex(-1),
mesh(mesh_)
{

}

Edge::Edge(const Edge& e):
onGenerator(e.onGenerator),
onCut(e.onCut),
isCuttable(e.isCuttable),
index(e.index),
halfEdgeIndex(e.halfEdgeIndex),
mesh(e.mesh)
{

}

HalfEdgeIter Edge::halfEdge() const
{
	return mesh->halfEdges.begin() + halfEdgeIndex;
}

void Edge::setHalfEdge(HalfEdgeCIter he)
{
	halfEdgeIndex = he->index;
}

void Edge::setMesh(Mesh *mesh_)
{
	mesh = mesh_;
}

bool Edge::onBoundary() const
{
	return halfEdge()->onBoundary || halfEdge()->flip()->onBoundary;
}

} // namespace bff
