#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#include <algorithm>
#include <array>
#include <iostream>

namespace bff {

using DenseMatrix = Eigen::MatrixXd;
using SparseMatrix = Eigen::SparseMatrix<double>;
using SparseSolver = Eigen::SimplicialLLT<SparseMatrix>;

inline auto diag(const DenseMatrix &m) { return m.asDiagonal(); }

inline DenseMatrix submatrix(const DenseMatrix &A, const std::vector<int> &is) { return A(is, 0); }

inline SparseMatrix submatrix(const SparseMatrix &A, const std::vector<int> &r) {
    if ((int)r.size() == A.rows() && (int)r.size() == A.cols()) return A;

    std::vector<int> map(A.rows(), -1);
    for (size_t i = 0; i < r.size(); ++i) {
        map[r[i]] = i;
    }

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> T;
    T.reserve(A.nonZeros());

    for (int j : r) {
        for (typename SparseMatrix::InnerIterator it(A, j); it; ++it) {
            int newRow = map[it.row()];
            if (newRow != -1) {
                T.emplace_back(newRow, map[j], it.value());
            }
        }
    }

    SparseMatrix out(r.size(), r.size());
    out.setFromTriplets(T.begin(), T.end());
    return out;
}

inline SparseMatrix submatrix(const SparseMatrix &A, const std::vector<int> &r, const std::vector<int> &c) {
    if ((int)r.size() == A.rows() && (int)c.size() == A.cols()) return A;

    // lookup maps
    std::vector<int> rowMap(A.rows(), -1);
    for (size_t i = 0; i < r.size(); ++i) {
        rowMap[r[i]] = i;
    }

    std::vector<int> colMap(A.cols(), -1);
    for (size_t j = 0; j < c.size(); ++j) {
        colMap[c[j]] = j;
    }

    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> T;
    T.reserve(A.nonZeros());

    for (int j : c) {
        for (typename SparseMatrix::InnerIterator it(A, j); it; ++it) {
            int newRow = rowMap[it.row()];
            if (newRow != -1) T.emplace_back(newRow, colMap[j], it.value());
        }
    }

    SparseMatrix out(r.size(), c.size());
    out.setFromTriplets(T.begin(), T.end());
    return out;
}

} // namespace bff
