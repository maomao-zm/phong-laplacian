//=============================================================================
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include "EnergySmoothing.h"
#include "PolyDiffGeo.h"
 
//=============================================================================

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

//=============================================================================

void EnergySmoothing::energy_smoothing(Scalar alpha) 
{
    if (!mesh_.n_vertices())
        return;

    //更新stiffness matrix,如果需要
    update_stiffness_matrix();

    const unsigned int nv = mesh_.n_vertices();
    

    SparseMatrix M(nv, nv);
    Eigen::MatrixX3d B(nv, 3);//coordinate

    for (auto v : mesh_.vertices())
    {
        B.row(v.idx()) = Eigen::Vector3d(mesh_.position(v));
  
    }

    setup_mass_matrix(mesh_, M, flag_);

    //更新光滑之后的点坐标
    
    SparseMatrix A = S_.transpose() * M * S_ + alpha * M;
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute(A);
    Eigen::MatrixXd X = solver.solve(alpha * M * B);

    if (solver.info() != Eigen::Success)
    {
        std::cerr << "SurfaceSmoothing: Could not solve linear system\n";
    }
    else
    {

        for (unsigned int i = 0; i < nv; ++i)
        {
            Vertex v(i);
            if (!mesh_.is_boundary(v))//mark
            {
                mesh_.position(v) = X.row(i);
            }
        }
            
    }

}


//=============================================================================

void EnergySmoothing::update_stiffness_matrix() 
{	
	if (mesh_.n_faces() != faces_ || mesh_.n_vertices() != vertices_
		|| clamp_ != clamp_cotan_)
    {
        vertices_ = mesh_.n_vertices();
        faces_ = mesh_.n_faces();
        clamp_ = clamp_cotan_;

        std::cout << "Stiffness matrix has been updated" << std::endl;
        setup_stiffness_matrix(mesh_, S_, flag_);
	}
}