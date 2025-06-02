#pragma once

#include <pmp/SurfaceMesh.h>
#include <Eigen/Sparse>

using namespace pmp;

class BoundarySmoothing
{
public:
    BoundarySmoothing(SurfaceMesh &mesh, int flag)
        : mesh_(mesh), vertices_(0), faces_(0), clamp_(false), flag_(flag)
    {
    }

    //! Perform Boundary Laplacian smoothing with alpha.
    void boundary_smoothing(Scalar alpha);

private:
    void update_stiffness_matrix();

private:
    SurfaceMesh &mesh_;
    Eigen::SparseMatrix<double> S_;
    unsigned int vertices_, faces_;
    bool clamp_;
    int flag_;
};
