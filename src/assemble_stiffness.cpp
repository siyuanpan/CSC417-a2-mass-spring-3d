#include <assemble_stiffness.h>
#include <d2V_spring_particle_particle_dq2.h>

void assemble_stiffness(Eigen::SparseMatrixd &K, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::VectorXd> qdot,
                        Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> E, Eigen::Ref<const Eigen::VectorXd> l0,
                        double k)
{
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplets;
    triplets.reserve(E.rows() * 9 * 4);
    for (int i = 0; i < E.rows(); ++i)
    {
        Eigen::Matrix66d H;
        auto v0 = E(i, 0);
        auto v1 = E(i, 1);
        auto q0 = V.row(E(i, 0)).transpose();
        auto q1 = V.row(E(i, 1)).transpose();
        d2V_spring_particle_particle_dq2(H, q0, q1, l0(i), k);
        for (int row = 0; row < 3; ++row)
        {
            for (int col = 0; col < 3; ++col)
            {
                auto val = -H(row, col);
                triplets.push_back(T(3 * v0 + row, 3 * v0 + col, val));
                triplets.push_back(T(3 * v0 + row, 3 * v1 + col, -val));
                triplets.push_back(T(3 * v1 + row, 3 * v0 + col, -val));
                triplets.push_back(T(3 * v1 + row, 3 * v1 + col, val));
            }
        }
    }

    K.resize(q.rows(), q.rows());
    K.setFromTriplets(triplets.begin(), triplets.end());
};