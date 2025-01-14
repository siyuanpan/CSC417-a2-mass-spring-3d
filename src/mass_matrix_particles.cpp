#include <mass_matrix_particles.h>

void mass_matrix_particles(Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::VectorXd> q, double mass)
{
    M.resize(q.rows(), q.rows());
    M.setIdentity();
    M *= mass;
}
