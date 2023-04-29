#pragma once

#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "bff/linear-algebra/Vector.h"

namespace bff {

class VertexAdjacencyMaps {
public:
	// constructs adjacency map
	void construct(int nV, const std::vector<int>& indices);

	// returns a single adjacent face and adjacent face count for vertex v
	std::pair<int, int> getAdjacentFaceCount(int v) const;

	// returns edge index corresponding to vertex entry (vi, vj)
	int getEdgeIndex(int vi, int vj) const;

	// returns edge count
	int getEdgeCount() const;

private:
	// members
	std::vector<int> data;
	std::vector<int> offsets;
	std::vector<std::pair<int, int>> faceCount;
};

class EdgeFaceAdjacencyMap {
public:
	// constructs adjacency map
	void construct(const VertexAdjacencyMaps& vertexAdjacency,
				   const std::vector<int>& indices);

	// returns adjacent face count for edge e
	int getAdjacentFaceCount(int e) const;

	// returns face index for edge e and 0 <= f <= getAdjacentFaceCount(e)
	int getAdjacentFaceIndex(int e, int f) const;

private:
	// members
	std::vector<int> data;
	std::vector<int> offsets;
};

class PolygonSoup {
public:
	// separates faces into components
	int separateFacesIntoComponents();

	// members
	std::vector<Vector> positions;
	std::vector<int> indices;
	std::vector<int> faceComponent;
	VertexAdjacencyMaps vertexAdjacency; // build after filling positions and indices
	EdgeFaceAdjacencyMap edgeFaceAdjacency; // build after constructing vertexAdjacency
};

} // namespace bff
