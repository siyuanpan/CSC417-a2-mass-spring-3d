#include <assemble_forces.h>
#include <dV_gravity_particle_dq.h>
#include <dV_spring_particle_particle_dq.h>
#include <iostream>

void assemble_forces(Eigen::VectorXd &f, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::VectorXd> qdot,
                     Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> E, Eigen::Ref<const Eigen::VectorXd> l0,
                     double mass, double k)
{
    f.resize(q.rows());
    f.setZero();
    // for (int i = 0; i < V.rows(); ++i) {
    //     dV_gravity_particle_dq(fi, mass, g);
    // }
    for (int i = 0; i < E.rows(); ++i)
    {
        auto q0 = V.row(E(i, 0)).transpose();
        auto q1 = V.row(E(i, 1)).transpose();
        Eigen::Vector6d fi;
        dV_spring_particle_particle_dq(fi, q0, q1, l0(i), k);
        f.segment<3>(3 * E(i, 0)) += -fi.segment<3>(0);
        f.segment<3>(3 * E(i, 1)) += -fi.segment<3>(3);
    }
    // std::cout << f << std::endl;
};