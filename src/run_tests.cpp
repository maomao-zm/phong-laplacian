//=============================================================================
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include <pmp/SurfaceMesh.h>
#include <pmp/algorithms/DifferentialGeometry.h>
#include <pmp/algorithms/SurfaceSubdivision.h>
#include <pmp/Timer.h>

#include "MeanCurvature.h"
#include "SphericalHarmonics.h"
#include "GeodesicsInHeat.h"
#include "PolyDiffGeo.h"

#include <Eigen/Sparse>
#include <iostream>
#include <fstream>

//=============================================================================

// to lump or not to lump the mass matrix?
bool LUMP_MASS_MATRIX = true;

//=============================================================================

using namespace pmp;
int flag_ = 2;

//=============================================================================

// dualize a polygon mesh.
// vertex positions of dual mesh are centroids of primal mesh.
void dualize(SurfaceMesh &mesh)
{
    SurfaceMesh dual;

    auto fvertex = mesh.add_face_property<Vertex>("f:vertex");
    for (auto f : mesh.faces())
    {
        fvertex[f] = dual.add_vertex(centroid(mesh, f));
    }

    for (auto v : mesh.vertices())
    {
        if (!mesh.is_boundary(v))
        {
            std::vector<Vertex> vertices;
            for (auto f : mesh.faces(v))
                vertices.push_back(fvertex[f]);
            dual.add_face(vertices);
        }
    }
    
    mesh = dual;
}

//----------------------------------------------------------------------------

// normalize all vertex position to norm=1.
// used to turn an almost spherical mesh into a spherical mesh.
void normalize(SurfaceMesh &mesh)
{
    for (auto v : mesh.vertices())
    {
        mesh.position(v) = normalize(mesh.position(v));
    }
}

//----------------------------------------------------------------------------

// compute curvature on a spherical mesh.
// compare to ground truth, i.e., 1
void test_curvatures(SurfaceMesh &mesh)
{
    Curvature curvature(mesh, flag_);
    curvature.compute();

    double rms = 0.0;
    for (auto v : mesh.vertices())
    {
        double c = curvature(v);
        rms += (c - 1.0) * (c - 1.0);
    }
    rms /= (double)mesh.n_vertices();
    rms = sqrt(rms);

    std::cout << "rms error = " << rms << std::endl;
}

//----------------------------------------------------------------------------

// compute geodesic distances from vertex 0 on a planar mesh.
// compare to ground truth, i.e., Euclidean distance.
void test_geodesics(SurfaceMesh &mesh)
{
    Vertex source(0);
    const Point p = mesh.position(source);

    GeodesicsInHeat heat(mesh, flag_);
    heat.compute_distance_from(source);

    double rms = 0.0;
    for (auto v : mesh.vertices())
    {
        double geodesic_dist = heat(v);
        double euclidean_dist = distance(p, mesh.position(v));
        rms += pow(geodesic_dist - euclidean_dist, 2.0);
    }
    rms /= (double)mesh.n_vertices();
    rms = sqrt(rms);

    std::cout << "rms error = " << rms << std::endl;
}

//----------------------------------------------------------------------------

// compute Eigenvectors of Laplacian on a spherical mesh.
// compare Eigenvectors to true spherical harmonics up to 9th band.
void test_sphericalH(SurfaceMesh &mesh)
{
    double error;
    double sum = 0.0;

    Eigen::VectorXd y(mesh.n_vertices());

    // setup mass and stiffness matrix
    Eigen::SparseMatrix<double> S, M;
    setup_stiffness_matrix(mesh, S);
    setup_mass_matrix(mesh, M);

    // pre-factorize mass matrix (might not be diagonal)
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(M);
    solver.factorize(M);

    for (int l=1; l<=8; l++)
    {
        double eval = -l * (l + 1);
        for (int m = -l; m <= l; m++)
        {
            for (auto v : mesh.vertices())
            {
                y(v.idx()) = spherical_harmonic(mesh.position(v), l, m);
            }
            y.normalize();

            Eigen::MatrixXd X = solver.solve(S * y);

            error = (y - 1.0 / eval * X).transpose() * M * (y - 1.0 / eval * X);

            sum += error;
        }
    }

    std::cout << "sh error = " << sum << std::endl;
}

//----------------------------------------------------------------------------

// measure time for constructing stiffness matrix
void timing_construction(SurfaceMesh mesh)
{
    const int trials = 20;

    pmp::Timer t;
    t.start();

    Eigen::SparseMatrix<double> S;
    for (int i = 0; i < trials; i++)
    {
        setup_stiffness_matrix(mesh, S);
    }

    t.stop();
    std::cout << "stiffness matrix construction: "
              << t.elapsed() / trials << "ms\n";
}

//----------------------------------------------------------------------------

// measure time for factoring/solving Poisson system
void timing_solution(SurfaceMesh mesh)
{
    auto points = mesh.vertex_property<Point>("v:point");

    // setup stiffness matrix
    Eigen::SparseMatrix<double> S;
    setup_stiffness_matrix(mesh, S);

    // setup mass matrix
    Eigen::SparseMatrix<double> M;
    setup_mass_matrix(mesh, M);

    // setup rhs 
    Eigen::MatrixXd B(mesh.n_vertices(), 3);
    for (auto v : mesh.vertices())
    {
        B.row(v.idx()) = static_cast<Eigen::Vector3d>(points[v]);
    }

    // setup implicit smoothing system
    Eigen::SparseMatrix<double> A;
    A = M - 0.1 * S;
    Eigen::MatrixXd M_B = M * B;

    
    static Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    const int trials = 10;


    // time for factoring matrix
    pmp::Timer t;
    t.start();
    for (int i = 0; i < trials; i++)
    {
        solver.compute(A);
    }
    t.stop();
    std::cout << "factorize matrix: " << t.elapsed() / trials
              << "ms\n";


    // time for back-substitution
    t.start();
    for (int i = 0; i < trials; i++)
    {
        Eigen::MatrixXd X = solver.solve(M_B);
    }
    t.stop();
    std::cout << "back-substitution: " << t.elapsed() / trials << "ms\n";
}

//----------------------------------------------------------------------------

/*
int main(int argc, char **argv)
{
    SurfaceMesh mesh;

    // which test to run
    int mytest = (argc > 1) ? atoi(argv[1]) : 0;

    // counter for tests
    int test = 1;

    // whether or not we lump the mass matrix
    lump_mass_matrix_ = LUMP_MASS_MATRIX;
    std::cout << "\nlump mass matrix = " << lump_mass_matrix_ << "\n\n";

   
    // compute mean curvature
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature: Hex sphere\n";
            mesh.read("../data/unit-sphere.off");
            dualize(mesh);
            normalize(mesh);
            test_curvatures(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature: Fine sphere\n";
            mesh.read("../data/unit-sphere.off");
            SurfaceSubdivision(mesh).catmull_clark();
            normalize(mesh);
            test_curvatures(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature: Regular sphere\n";
            mesh.read("../data/quad-sphere.off");
            normalize(mesh);
            test_curvatures(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Curvature : Noisy sphere\n";
            mesh.read("../data/noisy-sphere.off");
            normalize(mesh);
            test_curvatures(mesh);
        }
    }
    ++test;

    // compute spherical harmonics
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "SH: Hex sphere\n";
            mesh.read("../data/unit-sphere.off");
            dualize(mesh);
            normalize(mesh);
            test_sphericalH(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "SH: Fine sphere\n";
            mesh.read("../data/unit-sphere.off");
            SurfaceSubdivision(mesh).catmull_clark();
            normalize(mesh);
            test_sphericalH(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "SH: Regular sphere\n";
            mesh.read("../data/quad-sphere.off");
            normalize(mesh);
            test_sphericalH(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "SH: Noisy sphere\n";
            mesh.read("../data/noisy-sphere.off");
            normalize(mesh);
            test_sphericalH(mesh);
        }
    }
    ++test;

    // compute geodesic distance
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: Subdivided quad plane \n";
            mesh.read("../data/quad-plane.obj");
            SurfaceSubdivision(mesh).catmull_clark();
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: Quad plane\n";
            mesh.read("../data/quad-plane.obj");
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: L-plane\n";
            mesh.read("../data/L-plane.obj");
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist: Tetris-plane (non-starshaped)\n";
            mesh.read("../data/tetris.obj");
            test_geodesics(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Geodist:  Tetris-plane (non-convex)\n";
            mesh.read("../data/tetris_2.obj");
            test_geodesics(mesh);
        }
    }
    ++test;

    // measure timings
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Timings: Hexasphere\n";
            mesh.read("../data/Spheres/hexaSphere.off");
            timing_construction(mesh);
            timing_solution(mesh);
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "Timings: Fine Sphere\n";
            mesh.read("../data/Spheres/fineSphere.off");
            timing_construction(mesh);
            timing_solution(mesh);
        }
    }
    ++test;

    //measure stiffness and  mass matrix difference
    if (!mytest || mytest == test)
    {
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "difference: stiffness matrix\n";
            mesh.read("E:/data/demo.obj");
            Eigen::SparseMatrix<double> S;
            setup_stiffness_matrix(mesh, S);
            std::cout << S << std::endl;
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "difference: mass matrix\n";
            mesh.read("E:/data/demo.obj");
            Eigen::SparseMatrix<double> M;
            setup_mass_matrix(mesh, M);
            std::cout << M << std::endl;
        }
        {
            std::cout << "-----------------------------------------------\n";
            std::cout << "difference: laplacian matrix\n";
            mesh.read("E:/data/demo.obj");
            Eigen::SparseMatrix<double> S;
            setup_stiffness_matrix(mesh, S);
            Eigen::SparseMatrix<double> M;
            setup_mass_matrix(mesh, M);
            Eigen::MatrixXd L;
            Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(M);
            Eigen::SparseMatrix<double> I(M.rows(), M.cols());
            I.setIdentity();
            auto M_inv = solver.solve(I);
            L = M_inv * S;//M -1 * S
            std::cout << L << std::endl;
        }
    }   
    
}
*/

int main(int argc, char **argv) 
{
    std::cout << "input mesh path and out file name" << std::endl;
    SurfaceMesh mesh;
    std::string out = "D:/pycharm/Phong Laplacian/Data/";
    std::string outlapla = "D:/pycharm/Phong Laplacian/LaplacianData/";
    std::string outstiff = "D:/pycharm/Phong Laplacian/StiffData/";
    std::string outmass = "D:/pycharm/Phong Laplacian/MassData/";
    int flag;
    int vertex_source;//geodesic distances vertex source
    if (argc == 4)
    {   
        mesh.read(argv[1]);
        outlapla += argv[2];
        outstiff += argv[2];
        outmass += argv[2];
        flag = std::atoi(argv[3]);

        //print out laplacian coefficient n * n
        Eigen::SparseMatrix<double> S;
        setup_stiffness_matrix(mesh, S, flag);
        Eigen::SparseMatrix<double> M;
        setup_mass_matrix(mesh, M, flag);

        std::ofstream stiff_out(outstiff + ".txt");
        Eigen::MatrixXd tmp_S = (Eigen::MatrixXd)S;
        if (stiff_out)
        {
            for (int i = 0; i < mesh.n_vertices(); i++)
            {
                for (int j = 0; j < mesh.n_vertices(); j++)
                {
                    if (i == mesh.n_vertices() - 1 &&
                        j == mesh.n_vertices() - 1)
                    {
                        stiff_out << tmp_S(i, j);
                    }
                    else
                    {
                        stiff_out << tmp_S(i, j) << " ";
                    }
                }
            }
        }

        stiff_out.close();

        std::ofstream mass_out(outmass + ".txt");
        Eigen::MatrixXd tmp_M = (Eigen::MatrixXd)M;
        if (mass_out)
        {
            for (int i = 0; i < mesh.n_vertices(); i++)
            {
                for (int j = 0; j < mesh.n_vertices(); j++)
                {
                    if (i == mesh.n_vertices() - 1 &&
                        j == mesh.n_vertices() - 1)
                    {
                        mass_out << tmp_M(i, j);
                    }
                    else
                    {
                        mass_out << tmp_M(i, j) << " ";
                    }
                }
            }
        }

        mass_out.close();

        Eigen::MatrixXd L; //laplacian
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(M);
        Eigen::SparseMatrix<double> I(M.rows(), M.cols());
        I.setIdentity();
        auto M_inv = solver.solve(I);
        L = M_inv * S; //M -1 * S
        std::ofstream lapla_out(outlapla + ".txt");

        if (lapla_out)
        {
            for (int i = 0; i < mesh.n_vertices(); i++)
            {
                for (int j = 0; j < mesh.n_vertices(); j++)
                {
                    if (i == mesh.n_vertices() - 1 &&
                        j == mesh.n_vertices() - 1)
                    {
                        lapla_out << L(i, j);
                    }
                    else
                    {
                        lapla_out << L(i, j) << " ";
                    }
                }
            }
        }

        lapla_out.close();
        std::cout << "SUCCESS" << std::endl;
        return 1;
    }
    else if (argc == 5)
    {
        mesh.read(argv[1]);
        out += argv[2];
        flag = std::atoi(argv[3]);
        vertex_source = std::atoi(argv[4]);
    }
    else
    {
        std::cout << "input again, please" << std::endl;
        exit(1);
    }

    //print mean curvature 
    Curvature curvature(mesh, flag);
    curvature.compute();
    std::ofstream curv_out(out + "_curv_phong.txt");
    if (curv_out)//success
    {
        for (auto v : mesh.vertices())
        {
            if(v.idx() != mesh.n_vertices() - 1) 
                curv_out << curvature(v) << " ";
            else 
                curv_out << curvature(v);
        }
    }
    
    curv_out.close();

    //print geodesic distances
    Vertex source(vertex_source);
    //const Point p = mesh.position(source);

    GeodesicsInHeat heat(mesh, flag);
    heat.compute_distance_from(source);

    // find maximum distance
    Scalar maxdist(0);
    for (auto v : mesh.vertices())
    {
        if (heat(v) <= FLT_MAX)
        {
            maxdist = std::max(maxdist, heat(v));
        }
    }

    std::ofstream dist_out(out + "_dist_phong.txt");

    if (dist_out)
    {
        for (auto v : mesh.vertices())// relative distance 
        {
            if (v.idx() != mesh.n_vertices() - 1)
                dist_out << heat(v) / maxdist << " ";
            else
                dist_out << heat(v) / maxdist;
        }
    }
    
    dist_out.close();
    std::cout << "SUCCESS" << std::endl;


    
}


//=============================================================================
