//=============================================================================
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================
#pragma once
//=============================================================================

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/SurfaceNormals.h>


//=============================================================================

using namespace pmp;

//=============================================================================

// global setting: whether to clamp cotan negative weights to zero (default: false)
extern bool clamp_cotan_;

// global settings: whether to lump the mass matrix (default: true)
extern bool lump_mass_matrix_;

//=============================================================================

//void get_vvertex(const Eigen::MatrixXd &polygon, Eigen::VectorXd &vweights);

void pre_compute_normals(const SurfaceMesh &mesh);

//=============================================================================

void get_vvertex_hat(const Eigen::MatrixXd &polygon,
                     const std::vector<Normal> &normals,
                     const Eigen::VectorXd &vweights, Eigen::Vector3d &vvertex);

// compute (sparse) stiffness matrix for polygonal mesh
void setup_stiffness_matrix(const SurfaceMesh &mesh, Eigen::SparseMatrix<double> &S, int flag = 2);

// compute (sparse) mass matrix for polygonal mesh.
// global variable lump_mass_matrix_ determines whether the mass
// matrix will be lumped.
void setup_mass_matrix(const SurfaceMesh &mesh, Eigen::SparseMatrix<double> &M, int flag = 2);

// compute (dense) stiffness matrix for one polygon
void setup_polygon_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                    const std::vector<Normal> &normals,
                                    const Eigen::VectorXd &vweights, 
                                    Eigen::MatrixXd &L);

// subdivision once
void setup_triangle_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                     const std::vector<Normal> &normals,
                                     Eigen::MatrixXd &S);
// subdivision twice
void setup_triangle2_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                      const std::vector<Normal> &normals,
                                      Eigen::MatrixXd &S);

// subdivision three times
void setup_triangle3_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                      const std::vector<Normal> &normals,
                                      Eigen::MatrixXd &S);

// quadratic
void setup_quadratic_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                      const std::vector<Normal> &normals,
                                      Eigen::MatrixXd &S);

// compute (dense) mass matrix for one polygon
void setup_polygon_mass_matrix(const Eigen::MatrixXd &polygon,
                               const std::vector<Normal> &normals,
                               const Eigen::VectorXd &vweights,
                               Eigen::MatrixXd &M);

// subdivision 
void setup_triangle_mass_matrix(const Eigen::MatrixXd &polygon,
                               const std::vector<Normal> &normals,
                               Eigen::MatrixXd &M);

// subdivision twice
void setup_triangle2_mass_matrix(const Eigen::MatrixXd &polygon,
                                 const std::vector<Normal> &normals,
                                 Eigen::MatrixXd &M);

// subdivision three times
void setup_triangle3_mass_matrix(const Eigen::MatrixXd &polygon,
                                 const std::vector<Normal> &normals,
                                 Eigen::MatrixXd &M);

//quadratic
void setup_quadratic_mass_matrix(const Eigen::MatrixXd &polygon,
                                 const std::vector<Normal> &normals,
                                 Eigen::MatrixXd &M);

// lump matrix M to a diagonal matrix where the entries are the sums of the rows
void lump_matrix(Eigen::SparseMatrix<double> &M);

// compute the transformation matrix for an arbitrary mesh that inserts a chosen point per face
void setup_prolongation_matrix(const SurfaceMesh &mesh,
                               Eigen::SparseMatrix<double> &A);

// barycenter/my_centroid of mesh, computed as area-weighted mean of vertices.
Point area_weighted_centroid(const SurfaceMesh &mesh);

// computes the surface area of a polygon.
double face_area(const SurfaceMesh &mesh, Face f);

// compute the surface area of a polygon mesh (as sum of face areas).
Scalar mesh_area(const SurfaceMesh &mesh);

// Computes the gradient on a triangle formed by the points i, j and k.
Eigen::Vector3d gradient_hat_function(const Point& i, const Point& j, const Point& k);

// Computes the Gradient matrix for any given mesh.
void setup_gradient_matrix(const SurfaceMesh &mesh, Eigen::SparseMatrix<double> &G);

// Computes the Divergence matrix for any given mesh.
void setup_divergence_matrix(const SurfaceMesh &mesh,
                             Eigen::SparseMatrix<double> &Gt);

// Computes the diagonal mass matrix W needed for L = DWG.
void setup_gradient_mass_matrix(const SurfaceMesh &mesh,
                                Eigen::SparseMatrix<double> &M);

// Computes the squared triangle area minimizing points and its affine combination weights
// for each face and stores it in a prior defined property.
void setup_virtual_vertices(SurfaceMesh &mesh);

// compute the affine weights wrt the polygon vertices to form the virtual vertex
void compute_virtual_vertex(const Eigen::MatrixXd &poly,
                            Eigen::VectorXd &weights);

//=============================================================================
//barycentric Coordinate Interpolation
void barycentric_Coordinate_Interpolation(
    const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &centroid,
    Eigen::Vector3d &coord);
//=============================================================================