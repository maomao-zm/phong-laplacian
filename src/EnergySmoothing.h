#pragma once

#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/DifferentialGeometry.h>

#include <Eigen/Sparse>

//=============================================================================

using namespace pmp;

//=============================================================================

class EnergySmoothing
{
public:
    EnergySmoothing(SurfaceMesh &mesh, int flag)
        : mesh_(mesh), vertices_(0), faces_(0), clamp_(false), flag_(flag)
    {
    }

    //! Perform energy Laplacian smoothing with alpha.
    void energy_smoothing(Scalar alpha);

private:
    void update_stiffness_matrix();

private:
    SurfaceMesh &mesh_;
    Eigen::SparseMatrix<double> S_;
    unsigned int vertices_, faces_;
    bool clamp_;
    int flag_;
};

//=============================================================================
