#include "PolyDiffGeo.h"
#include "BoundarySmoothing.h"

#include <pmp/algorithms/DifferentialGeometry.h>

using SparseMatrix = Eigen::SparseMatrix<double>;

void BoundarySmoothing::boundary_smoothing(Scalar alpha)
{
    //Eigen::MatrixXd B(nv, 3); 
    /*
    for (auto v : mesh_.vertices())
    {
        B.row(v.idx()) = Eigen::Vector3d(mesh_.position(v));
    }
    
    Eigen::MatrixXd diff = L * B;
    
    unsigned int count = 0;

    for (unsigned int i = 0; i < nv; ++i)
    {
        Vertex v(i);
        if (!mesh_.is_boundary(v)) //mark
        {
            mesh_.position(v) -= alpha * normalize(diff.row(i));
        }
        else
            count++;
    }

    std::cout << "boundary points number: "<< count << std::endl;
    */

    
    if (!mesh_.n_vertices())
        return;

    unsigned int bv = 0;//boundary points number
    std::vector<int> b_v;//boundary points index

    for (auto v : mesh_.vertices())
    {
        if (mesh_.is_boundary(v))
        {
            b_v.push_back(v.idx());
            bv++;
        }
    }
    
    if (bv == 0)
    {
        std::cout << " no boundary" << std::endl;
        return;
    }
    else
        std::cout << "boundary points number: " << bv << std::endl;

    // update stiffness matrix (if required)
    update_stiffness_matrix();

    const unsigned int nv = mesh_.n_vertices();
    SparseMatrix M;

    setup_mass_matrix(mesh_, M, flag_);

    Eigen::MatrixXd L;
    L = ((Eigen::MatrixXd)M).inverse() * (Eigen::MatrixXd)S_; //M -1 * S laplacian

    Eigen::MatrixXd B(nv, 3); 
    for (auto v : mesh_.vertices())
    {
        B.row(v.idx()) = Eigen::Vector3d(mesh_.position(v));
    }

    Eigen::MatrixXd b = alpha * L * B;//nv * 3
    b.conservativeResize(nv + bv, 3);

    for (unsigned int i = 0; i < bv; i++)
    {
        b.row(nv + i) = 100.0 * Eigen::Vector3d(mesh_.position(Vertex(b_v[i])));//boundary points
    }

    L.conservativeResize(nv + bv, nv);
    Eigen::MatrixXd A = Eigen::MatrixXd(L);//convert to dense matrix to operate

    for (unsigned int i = 0; i < bv; i++)
    {
        A(nv + i, b_v[i]) = 100.0;
    }

    SparseMatrix a = A.sparseView();//convert to sparse matrix to solve;
    //using normal equation 
    Eigen::MatrixXd X;
    Eigen::SimplicialLDLT<SparseMatrix> solver;
    solver.compute(a.transpose() * a);
    X = solver.solve(a.transpose() * b);

    if (solver.info() != Eigen::Success)
    {
        std::cerr << "SurfaceSmoothing: Could not solve linear system\n";
    }
    else
    {
        // copy solution
        for (unsigned int i = 0; i < nv; ++i)
        {
            Vertex v(i);
            //std::cout << X.row(i) << std::endl;
            mesh_.position(v) = X.row(i);
        }
    }
    
}

void BoundarySmoothing::update_stiffness_matrix()
{
    if (mesh_.n_faces() != faces_ || mesh_.n_vertices() != vertices_ ||
        clamp_ != clamp_cotan_)
    {
        vertices_ = mesh_.n_vertices();
        faces_ = mesh_.n_faces();
        clamp_ = clamp_cotan_;

        std::cout << "Stiffness matrix has been updated" << std::endl;
        setup_stiffness_matrix(mesh_, S_, flag_);
    }
}