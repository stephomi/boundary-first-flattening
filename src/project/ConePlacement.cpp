#include "bff/project/ConePlacement.h"
#include <limits>

namespace bff {

int ConePlacement::initializeConeSet(const DenseMatrix& K, const WedgeData<int>& index,
									 VertexData<uint8_t>& isCone, Mesh& mesh)
{
	int cones = 0;
	if (mesh.boundaries.size() > 0) {
		// designate all boundary vertices as cone vertices
		for (WedgeCIter w: mesh.cutBoundary()) {
			isCone[w->vertex()] = 1;
			cones++;
		}

	} else if (mesh.eulerCharacteristic() != 0) {
		// surface is closed, select the vertex with largest curvature as a cone
		// singularity if the euler characteristic is greater than 0 and vice versa
		VertexCIter cone;
		double curvature = std::numeric_limits<double>::infinity();
		if (mesh.eulerCharacteristic() > 0) curvature *= -1;

		for (VertexCIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++) {
			if (v->insideHole()) continue;
			int i = index[v->wedge()];

			bool isCandidateCone = mesh.eulerCharacteristic() > 0 ? curvature < K(i) : curvature > K(i);
			if (isCandidateCone) {
				curvature = K(i);
				cone = v;
			}
		}

		isCone[cone] = 1;
		cones++;
	}

	return cones;
}

void ConePlacement::separateConeIndices(std::vector<int>& s, std::vector<int>& n,
										const VertexData<uint8_t>& isCone,
										const WedgeData<int>& index, const Mesh& mesh,
										bool ignoreBoundary)
{
	// initialize
	s.clear();
	n.clear();

	// collect cone and non-cone indices
	for (VertexCIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++) {
		int i = index[v->wedge()];

		if (isCone[v] == 1) {
			bool onBoundary = v->onBoundary();
			if (!onBoundary || (onBoundary && !ignoreBoundary)) {
				s.emplace_back(i);
			}

		} else {
			n.emplace_back(i);
		}
	}
}

bool ConePlacement::computeTargetAngles(DenseMatrix& C, const DenseMatrix& K,
										const SparseMatrix& A,
										const VertexData<uint8_t>& isCone,
										const WedgeData<int>& index, const Mesh& mesh)
{
	// collect cone and non-cone indices
	std::vector<int> s, n;
	separateConeIndices(s, n, isCone, index, mesh);

	int S = (int)s.size();
	if (S > 0) {
		// initialize cone angles
		DenseMatrix Ks = submatrix(K, s);
		for (int i = 0; i < S; i++) C(s[i],0) = Ks(i,0);

		if (n.size() > 0) {
			// extract submatrices
			SparseMatrix Ann = submatrix(A, n);
			SparseMatrix Ans = submatrix(A, n, s);
			DenseMatrix Kn = submatrix(K, n);

			// Factorise
			SparseSolver solver;
			solver.analyzePattern(Ann);
			solver.factorize(Ann);
			if (solver.info() != Eigen::Success) return false;
			// compute target curvatures
			for (int i = 0; i < S; i++) {
				int I = s[i];

				// solve LGn = δn
				DenseMatrix Gn, Gs(S, 1);
				Gs(i,0) = 1;
				DenseMatrix delta = -(Ans*Gs);
				Gn = solver.solve(delta);
				if (solver.info() != Eigen::Success) return false;

				// Cs = Ks + Gn^T Kn
				C(I,0) += (Gn.transpose()*Kn)(0);
			}
		}
	}

	return true;
}

void ConePlacement::computeTargetAngles(DenseMatrix& C, const DenseMatrix& u,
										const DenseMatrix& K, const DenseMatrix& k,
										const SparseMatrix& A, const VertexData<uint8_t>& isCone,
										const WedgeData<int>& index, Mesh& mesh)
{
	// collect cone, non-cone and boundary indices
	std::vector<int> s, n, b;
	separateConeIndices(s, n, isCone, index, mesh, true);
	for (WedgeCIter w: mesh.cutBoundary()) b.emplace_back(index[w]);

	int I = (int)K.rows();
	int B = (int)k.rows();
	int S = (int)s.size();

	if (S > 0) {
		// initialize cone angles
		DenseMatrix Ks = submatrix(K, s);
		for (int i = 0; i < S; i++) C(s[i]) = Ks(i);
		for (int i = 0; i < B; i++) C(b[i]) = k(b[i] - I);

		if (n.size() > 0) {
			// extract submatrices
			SparseMatrix Asn = submatrix(A, s, n);
			SparseMatrix Abn = submatrix(A, b, n);
			DenseMatrix un = submatrix(u, n);

			// compute interior cone angles
			DenseMatrix Cs = -(Asn*un);
			for (int i = 0; i < S; i++) C(s[i]) -= Cs(i);

			// compute boundary cone angles
			DenseMatrix h = -(Abn*un);
			for (int i = 0; i < B; i++) C(b[i]) -= h(i);
		}
	}
}

bool ConePlacement::addConeWithLargestScaleFactor(VertexData<uint8_t>& isCone,
												  const DenseMatrix u,
												  const WedgeData<int>& index,
												  const Mesh& mesh)
{
	VertexCIter cone;
	double maxU = -std::numeric_limits<double>::infinity();

	for (VertexCIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++) {
		if (!v->onBoundary() && !v->insideHole() && !isCone[v]) {
			int i = index[v->wedge()];

			double absU = std::abs(u(i));
			if (maxU < absU) {
				maxU = absU;
				cone = v;
			}
		}
	}

	if (std::isinf(maxU) || std::isnan(maxU)) {
		for (VertexCIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++) {
			isCone[v] = 0;
		}

		return false;
	}

	isCone[cone] = 1;
	return true;
}

bool ConePlacement::computeScaleFactors(DenseMatrix& u, const DenseMatrix& K,
										const SparseMatrix& A, const VertexData<uint8_t>& isCone,
										const WedgeData<int>& index, const Mesh& mesh)
{
	// collect cone and non-cone indices
	std::vector<int> s, n;
	separateConeIndices(s, n, isCone, index, mesh, true);

	// initialize scale factors
	u = DenseMatrix(K.rows(), 1);

	if (n.size() > 0) {
		// extract submatrices
		DenseMatrix Kn = submatrix(-K, n);
		SparseMatrix Ann = submatrix(A, n);

		// compute scale factors
		SparseSolver solver;
		solver.analyzePattern(Ann);
		solver.factorize(Ann);
		if (solver.info() != Eigen::Success) return false;
		DenseMatrix un = solver.solve(Kn);
		if (solver.info() != Eigen::Success) return false;

		// collect scale factors
		for (int i = 0; i < (int)n.size(); i++) {
			u(n[i],0) = un(i,0);
		}
	}

	return true;
}

bool ConePlacement::useCpmsStrategy(int S, VertexData<uint8_t>& isCone,
									DenseMatrix& C, const DenseMatrix& K,
									SparseMatrix& A, const WedgeData<int>& index,
									Mesh& mesh)
{
	// initialize cone set
	int cones = initializeConeSet(K, index, isCone, mesh);
	if (mesh.boundaries.size() == 0) S -= cones;

	// compute target angles
	if (!computeTargetAngles(C, K, A, isCone, index, mesh)) return false;

	for (int i = 0; i < S; i++) {
		// compute scale factors
		DenseMatrix rhs = -(K - C);
		SparseSolver solver;
		solver.analyzePattern(A);
		solver.factorize(A);
		if (solver.info() != Eigen::Success) return false;
		DenseMatrix u = solver.solve(rhs);
		if (solver.info() != Eigen::Success) return false;

		// add vertex with maximum (abs.) scaling to cone set
		if (!addConeWithLargestScaleFactor(isCone, u, index, mesh)) {
			return false;
		}

		// compute target angles
		if (!computeTargetAngles(C, K, A, isCone, index, mesh)) return false;
	}

	return true;
}

bool ConePlacement::useCetmStrategy(int S, VertexData<uint8_t>& isCone,
									DenseMatrix& C, const DenseMatrix& K,
									const DenseMatrix& k, const SparseMatrix& A,
									const WedgeData<int>& index, Mesh& mesh)
{
	// initialize cone set
	int cones = initializeConeSet(K, index, isCone, mesh);
	if (mesh.boundaries.size() == 0) S -= cones;

	// compute scale factors
	DenseMatrix u;
	if (!computeScaleFactors(u, K, A, isCone, index, mesh)) return false;

	for (int i = 0; i < S; i++) {
		// add vertex with maximum (abs.) scaling to cone set
		if (!addConeWithLargestScaleFactor(isCone, u, index, mesh)) {
			return false;
		}

		// compute scale factors
		if (!computeScaleFactors(u, K, A, isCone, index, mesh)) return false;
	}

	// compute cone angles
	computeTargetAngles(C, u, K, k, A, isCone, index, mesh);

	return true;
}

void ConePlacement::normalizeAngles(DenseMatrix& C, double normalizationFactor)
{
	double sum = C.sum();
	if (sum > 1e-8) C *= (normalizationFactor/sum);
}

ConePlacement::ErrorCode ConePlacement::findConesAndPrescribeAngles(
											int S, std::vector<VertexIter>& cones,
											DenseMatrix& coneAngles,
											std::shared_ptr<BFFData> data, Mesh& mesh)
{
	VertexData<uint8_t> isCone(mesh, 0);
	DenseMatrix C = DenseMatrix::Zero(data->N, 1);
	DenseMatrix K = vcat(data->K, data->k);
	bool success = true;

	if ((success = useCetmStrategy(S, isCone, C, data->K, data->k, data->A, data->index, mesh))) {
		// set cones
		cones.reserve(S);
		for (VertexIter v = mesh.vertices.begin(); v != mesh.vertices.end(); v++) {
			if (!v->onBoundary() && isCone[v] == 1) {
				cones.emplace_back(v);
			}
		}

		// normalize angles
		normalizeAngles(C, 2*M_PI*mesh.eulerCharacteristic());
	}

	// set cone angles
	coneAngles = C.topRows(data->iN);
	return success ? ConePlacement::ErrorCode::ok :
					 ConePlacement::ErrorCode::factorizationFailed;
}

} // namespace bff
