#pragma once

#include <pmp/SurfaceMesh.h>
#include <Eigen/Sparse>

using namespace pmp;

class Deformation
{
public:
    Deformation(SurfaceMesh &mesh, int flag)
        : mesh_(mesh), vertices_(0), faces_(0), clamp_(false), flag_(flag)
    {
    }

    //! Perform deformation via Laplacian editing.

    void do_deform(Eigen::Vector3d hPosition, unsigned int hIdx, unsigned int nRoi,
                   std::vector<int> roiIndices);

private:
    void update_stiffness_matrix();

private:
    SurfaceMesh &mesh_;
    Eigen::SparseMatrix<double> S_;
    unsigned int vertices_, faces_;
    bool clamp_;
    int flag_;
};
