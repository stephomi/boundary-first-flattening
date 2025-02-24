#pragma once

#include "bff/mesh/Types.h"
#include "bff/mesh/Spline.h"

namespace bff {

class Corner {
public:
	// constructor
	Corner(Mesh *mesh);

	// copy constructor
	Corner(const Corner& c);

	// returns the halfedge opposite to this corner
	HalfEdgeIter halfEdge() const;

	// sets halfedge
	void setHalfEdge(HalfEdgeCIter he);

	// returns the vertex associated with this corner
	VertexIter vertex() const;

	// returns the face associated with this corner
	FaceIter face() const;

	// returns the next corner (in CCW order) associated with this corner's face
	CornerIter next() const;

	// returns the previous corner (in CCW order) associated with this corner's face
	CornerIter prev() const;

	// returns the next wedge (in CCW order) along the (possibly cut) boundary
	WedgeIter nextWedge() const;

	// sets mesh
	void setMesh(Mesh *mesh);

	// checks if this corner is real
	bool isReal() const;

	// uv coordinates
	Vector uv;

	// flag to indicate whether this corner is contained in a face that is incident
	// to the north pole of a stereographic projection from the disk to a sphere
	bool inNorthPoleVicinity;

	// spline handle for direct editing
	KnotIter knot;

	// id between 0 and |C|-1
	int index;

private:
	// index of the halfedge associated with this corner
	int halfEdgeIndex;

	// pointer to mesh this corner belongs to
	Mesh *mesh;
};

} // namespace bff
