
#include "PolyDiffGeo.h"
#include "Deformation.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>

using SparseMatrix = Eigen::SparseMatrix<double>; 
   
void Deformation::do_deform(Eigen::Vector3d hPosition, unsigned int hIdx,
    unsigned int nRoi,
    std::vector<int> roiIndices)
{
    if (!mesh_.n_vertices())
        return;

    // update stiffness matrix (if required)
    update_stiffness_matrix();

    const unsigned int nv = mesh_.n_vertices();
    SparseMatrix L,M;

    setup_mass_matrix(mesh_, M, flag_);
    L = M * S_;
    //Eigen::MatrixXd L;
    //L = ((Eigen::MatrixXd)M).inverse() * (Eigen::MatrixXd)S_; //M -1 * S laplacian

    Eigen::MatrixXd B(nv, 3);
    for (auto v : mesh_.vertices())
    {
        B.row(v.idx()) = Eigen::Vector3d(mesh_.position(v));
    }

    Eigen::MatrixXd b = L * B; //nv * 3
    b.conservativeResize(nv + nRoi, 3);

    for (unsigned int i = 0; i < nRoi; i++)
    {
        if (roiIndices[i] == hIdx)
            b.row(nv + i) = 100.0 * hPosition;
        else
            b.row(nv + i) = 100.0 * Eigen::Vector3d(mesh_.position(Vertex(roiIndices[i]))); //ROI points
    }

    L.conservativeResize(nv + nRoi, nv);
    Eigen::MatrixXd A = Eigen::MatrixXd(L); //convert to dense matrix to operate
    
    for (unsigned int i = 0; i < nRoi; i++)
    {
        A(nv + i, roiIndices[i]) = 100.0;
    }

    SparseMatrix a = A.sparseView(); //convert to sparse matrix to solve;
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
    
    
void Deformation::update_stiffness_matrix()
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

