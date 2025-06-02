//=============================================================================
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include "PolyDiffGeo.h"
#include <pmp/algorithms/DifferentialGeometry.h>
#include <pmp/algorithms/SurfaceTriangulation.h>
#include <pmp/algorithms/SurfaceNormals.h>

#include "VertexNormal.h"

//=============================================================================

using namespace std;
using namespace pmp;

using SparseMatrix = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

//=============================================================================

const double eps = 1e-10;
bool clamp_cotan_ = false;
bool lump_mass_matrix_ = true;
const double alpha = (double)3 / 4;


std::vector<Normal> pre_normals;
bool is_precompute = false;


void pre_compute_normals(const SurfaceMesh &mesh)
{   
    pre_normals.resize(mesh.n_vertices());

    if (is_precompute)
        return;

    for (auto v : mesh.vertices())
    {
        pre_normals[v.idx()] = pmp::SurfaceNormals::compute_vertex_normal(mesh, v);
    }

    is_precompute = true;
    std::cout << "pre compute normals" << std::endl;
}
//=============================================================================

void compute_virtual_vertex(const Eigen::MatrixXd &polygon,
                 Eigen::VectorXd &vweights)
{
    //由于质心求法，因此，各权重相等
    int n = polygon.rows();
    vweights.resize(n);

    //vweights(0) = 0.5;
    //vweights(1) = 0.5;
   
    for (int i = 0; i < n; i++)
    {
       vweights(i) = (double)1 / n;
    }
    

    /*
    //重心坐标插值
    Eigen::MatrixXd tmp; //3*2
    tmp.resize(3, 2);
    tmp.col(0) = (Eigen::Vector3d)((polygon.row(1) - polygon.row(0)).transpose());//AB
    tmp.col(1) = (Eigen::Vector3d)((polygon.row(2) - polygon.row(0)).transpose());//AC


    //求解一个非齐次线性方程组
    Eigen::Vector2d uv = 
        tmp.colPivHouseholderQr().solve(centroid - (Eigen::Vector3d)polygon.row(0));
    vweights(0) = 1 - uv(0) - uv(1); //w
    vweights(1) = uv(0);             //u
    vweights(2) = uv(1);             //v
    */
}


//=============================================================================

void get_vvertex_hat(const Eigen::MatrixXd &polygon,
                     const std::vector<Normal> &normals,
                     const Eigen::VectorXd &vweights, 
                     Eigen::Vector3d &vvertex)
{
    Eigen::Vector3d p_hat; //p*
    p_hat.setZero();
    int n = (int)polygon.rows();

    for (int i = 0; i < n; i++)
    {	
		//在此处修改polygon,为了匹配vweights
		/*polygon.row(i) = (vvertex - ((vvertex - (Eigen::Vector3d)(polygon.row(i)).transpose())
                                         .transpose()
                                         .dot((Eigen::Vector3d)normals[i])) *
                                        ((Eigen::Vector3d)normals[i]));
		p_hat += vweights(i) * polygon.row(i);*/
        p_hat +=
            vweights(i) *
            (vvertex - ((vvertex - (Eigen::Vector3d)(polygon.row(i)).transpose())
                                         .transpose()
                                         .dot((Eigen::Vector3d)normals[i])) *
                                        ((Eigen::Vector3d)normals[i]));

    }

    //形状因子为固定为3/4
    vvertex = (1 - alpha) * vvertex + alpha * p_hat;

    //vvertex = -2 * (p_hat - vvertex) + vvertex;
}

//=============================================================================

void setup_stiffness_matrix(const SurfaceMesh &mesh, Eigen::SparseMatrix<double> &S, int flag)
{
    const int nv = mesh.n_vertices();

    std::vector<Vertex> vertices; // polygon vertices
    Eigen::MatrixXd polygon;      // positions of polygon vertices
    std::vector<Normal> normals; // polygon normals
    Eigen::VectorXd weights;      // affine weights of virtual vertex
    Eigen::MatrixXd Si;           // local stiffness matrix

    std::vector<Eigen::Triplet<double>> trip;
    
    for (Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        const int n=vertices.size();

        // collect their positions
        polygon.resize(n, 3);
        for (int i=0; i<n; ++i)
        {
            polygon.row(i) = (Eigen::Vector3d) mesh.position(vertices[i]);
            
            pre_compute_normals(mesh);
            normals.push_back(pre_normals[vertices[i].idx()]);
            //normals.push_back(pmp::SurfaceNormals::compute_vertex_normal(mesh, vertices[i]));
        }

        // compute virtual vertex, setup local stiffness matrix
        if (flag == 0)
        {
            compute_virtual_vertex(polygon, weights);//Centroid
            setup_polygon_stiffness_matrix(polygon, normals, weights, Si);//Centroid
        }
        else if (flag == 1)
            setup_triangle_stiffness_matrix(polygon, normals, Si); //subdivision once 
        else if (flag == 2)
            setup_triangle2_stiffness_matrix(polygon, normals, Si);//subdivison twice
        else if (flag == 3)
            setup_triangle3_stiffness_matrix(polygon, normals, Si); //subdivison three times
        else 
            setup_quadratic_stiffness_matrix(polygon, normals, Si); //quadratic

        // assemble to global stiffness matrix
        for (int j=0; j<n; ++j)
        {
            for (int k=0; k<n; ++k)
            {
                trip.emplace_back(vertices[k].idx(), vertices[j].idx(), -Si(k, j));
            }
        }
    }

    // build sparse matrix from triplets
    S.resize(nv, nv);
    S.setFromTriplets(trip.begin(), trip.end());
}

//----------------------------------------------------------------------------------


void setup_polygon_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                    const std::vector<Normal> &normals,
                                    const Eigen::VectorXd &vweights, 
                                    Eigen::MatrixXd &S)
{
    const int n = (int)polygon.rows();
    S.resize(n, n);
    S.setZero();
    
    // compute position of virtual vertex
    Eigen::Vector3d vvertex = polygon.transpose() * vweights;
    // phong tessellation 
    get_vvertex_hat(polygon, normals, vweights, vvertex);

    Eigen::VectorXd ln(n + 1);
    ln.setZero();

    double l[3], l2[3];

    for (int i = 0; i < n; ++i)
    {
        const int i1 = (i + 1) % n;

        l2[2] = (polygon.row(i) - polygon.row(i1)).squaredNorm();
        l2[0] = (polygon.row(i1) - vvertex.transpose()).squaredNorm();
        l2[1] = (polygon.row(i) - vvertex.transpose()).squaredNorm();

        l[0] = sqrt(l2[0]);
        l[1] = sqrt(l2[1]);
        l[2] = sqrt(l2[2]);

        const double arg = (l[0] + (l[1] + l[2])) * (l[2] - (l[0] - l[1])) *
                           (l[2] + (l[0] - l[1])) * (l[0] + (l[1] - l[2]));
        const double area = 0.5 * sqrt(arg);
        if (area > 1e-7)
        {
            l[0] = 0.25 * (l2[1] + l2[2] - l2[0]) / area;
            l[1] = 0.25 * (l2[2] + l2[0] - l2[1]) / area;
            l[2] = 0.25 * (l2[0] + l2[1] - l2[2]) / area;

            S(i1, i1) += l[0];
            S(i, i) += l[1];
            S(i1, i) -= l[2];
            S(i, i1) -= l[2];
            S(i, i) += l[2];
            S(i1, i1) += l[2];

            ln(i1) -= l[0];
            ln(i) -= l[1];
            ln(n) += l[0] + l[1];
        }
    }

    // sandwiching with (local) restriction and prolongation matrix
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            S(i, j) += vweights(i) * ln(j) + vweights(j) * ln(i) + vweights(i) * vweights(j) * ln(n);
}

//----------------------------------------------------------------------------------

void setup_triangle_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                     const std::vector<Normal> &normals,
                                     Eigen::MatrixXd &S)
{
    const int n = (int)polygon.rows();// n == 3
    if (n != 3)
        return;
    S.resize(3, 3);

    Eigen::MatrixXd S_F, P, V;
    S_F.resize(6, 6);
    S_F.setZero();
    P.resize(6, 3);
    V.resize(6, 3);

    //construct the prolongation matrix;
    P.row(0) = Eigen::Vector3d(1., 0., 0.);
    P.row(1) = Eigen::Vector3d(0., 1., 0.);
    P.row(2) = Eigen::Vector3d(0., 0., 1.);
    P.row(3) = Eigen::Vector3d(0.5, 0.5, 0.);
    P.row(4) = Eigen::Vector3d(0., 0.5, 0.5);
    P.row(5) = Eigen::Vector3d(0.5, 0., 0.5);

    //construct the vertex matrix;
    //V.row(0) = polygon.row(0);
    //V.row(1) = polygon.row(1);
    //V.row(2) = polygon.row(2);
    
    // compute position of virtual vertex
    Eigen::Vector3d vvertex =
        polygon.transpose() * Eigen::Vector3d(0.5, 0.5, 0.);
    // phong tessellation 
    get_vvertex_hat(polygon, normals, Eigen::Vector3d(0.5, 0.5, 0.), vvertex);
    V.row(3) = vvertex;

    // compute position of virtual vertex
    vvertex = polygon.transpose() * Eigen::Vector3d(0., 0.5, 0.5);
    // phong tessellation
    get_vvertex_hat(polygon, normals, Eigen::Vector3d(0., 0.5, 0.5), vvertex);
    V.row(4) = vvertex;

    // compute position of virtual vertex
    vvertex = polygon.transpose() * Eigen::Vector3d(0.5, 0., 0.5);
    // phong tessellation
    get_vvertex_hat(polygon, normals, Eigen::Vector3d(0.5, 0., 0.5), vvertex);
    V.row(5) = vvertex;
    
    //construct the vertex matrix;
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    //construct refine stiffness matrix;
    S_F(0, 3) = pmp::cotan(V.row(3) - V.row(5), V.row(0) - V.row(5)) / 2;
    S_F(0, 5) = pmp::cotan(V.row(5) - V.row(3), V.row(0) - V.row(3)) / 2;

    S_F(1, 3) = pmp::cotan(V.row(1) - V.row(4), V.row(3) - V.row(4)) / 2;
    S_F(1, 4) = pmp::cotan(V.row(1) - V.row(3), V.row(4) - V.row(3)) / 2;

    S_F(2, 4) = pmp::cotan(V.row(4) - V.row(5), V.row(2) - V.row(5)) / 2;
    S_F(2, 5) = pmp::cotan(V.row(5) - V.row(4), V.row(2) - V.row(4)) / 2;

    S_F(3, 4) = (pmp::cotan(V.row(3) - V.row(1), V.row(4) - V.row(1)) +
                pmp::cotan(V.row(3) - V.row(5), V.row(4) - V.row(5))) / 2;

    S_F(3, 5) = (pmp::cotan(V.row(3) - V.row(0), V.row(5) - V.row(0)) +
                 pmp::cotan(V.row(3) - V.row(4), V.row(5) - V.row(4))) / 2;

    S_F(4, 5) = (pmp::cotan(V.row(4) - V.row(2), V.row(5) - V.row(2)) +
                 pmp::cotan(V.row(4) - V.row(3), V.row(5) - V.row(3))) / 2;

    for (int j = 0; j < 6; j++)//transpose 
    {   
        for (int i = j + 1; i < 6; i++)
        {
            S_F(i, j) = S_F(j, i);
        }
    }

    for (int i = 0; i < 6; i++)
    {
        double w = 0;
        for (int j = 0; j < 6; j++)
        {
            w += S_F(i, j);
        }
        S_F(i, i) = -w;
    }

    // sandwiching with (local) restriction and prolongation matrix
    S = - (P.transpose() * S_F * P);
}

//----------------------------------------------------------------------------------

void setup_triangle2_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                     const std::vector<Normal> &normals,
                                     Eigen::MatrixXd &S)
{
    const int n = (int)polygon.rows(); // n == 3
    if (n != 3)
        return;
    S.resize(3, 3);

    Eigen::MatrixXd S_F, P, V;
    S_F.resize(15, 15);
    S_F.setZero();
    P.resize(15, 3);
    V.resize(15, 3);

    //construct the prolongation matrix;
    P.row(0) = Eigen::Vector3d(1., 0., 0.);
    P.row(1) = Eigen::Vector3d(0., 1., 0.);
    P.row(2) = Eigen::Vector3d(0., 0., 1.);

    P.row(3) = Eigen::Vector3d(0.75, 0.25, 0.);
    P.row(4) = Eigen::Vector3d(0.5, 0.5, 0.);
    P.row(5) = Eigen::Vector3d(0.25, 0.75, 0.);

    P.row(6) = Eigen::Vector3d(0., 0.75, 0.25);
    P.row(7) = Eigen::Vector3d(0., 0.5, 0.5);
    P.row(8) = Eigen::Vector3d(0., 0.25, 0.75);

    P.row(9) = Eigen::Vector3d(0.25, 0., 0.75);
    P.row(10) = Eigen::Vector3d(0.5, 0., 0.5);
    P.row(11) = Eigen::Vector3d(0.75, 0., 0.25);

    P.row(12) = Eigen::Vector3d(0.5, 0.25, 0.25);
    P.row(13) = Eigen::Vector3d(0.25, 0.5, 0.25);
    P.row(14) = Eigen::Vector3d(0.25, 0.25, 0.5);

    //construct the vertex matrix;
    //V.row(0) = polygon.row(0);
    //V.row(1) = polygon.row(1);
    //V.row(2) = polygon.row(2);
    
    // compute position of virtual vertex
    // phong tessellation
    Eigen::Vector3d vvertex;
    for (int i = 3; i < 15; i++)
    {
        vvertex = polygon.transpose() * Eigen::Vector3d(P.row(i));
        get_vvertex_hat(polygon, normals, Eigen::Vector3d(P.row(i)), vvertex);
        V.row(i) = vvertex;
    }

    //construct the vertex matrix;
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    //construct refine stiffness matrix;
    // 30 edges
    S_F(0, 3) = pmp::cotan(V.row(3) - V.row(11), V.row(0) - V.row(11)) / 2;
    S_F(0, 11) = pmp::cotan(V.row(11) - V.row(3), V.row(0) - V.row(3)) / 2;

    S_F(1, 5) = pmp::cotan(V.row(1) - V.row(6), V.row(5) - V.row(6)) / 2;
    S_F(1, 6) = pmp::cotan(V.row(1) - V.row(5), V.row(6) - V.row(5)) / 2;

    S_F(2, 8) = pmp::cotan(V.row(2) - V.row(9), V.row(8) - V.row(9)) / 2;
    S_F(2, 9) = pmp::cotan(V.row(2) - V.row(8), V.row(9) - V.row(8)) / 2;

    S_F(3, 4) = pmp::cotan(V.row(3) - V.row(12), V.row(4) - V.row(12)) / 2;
    S_F(3, 11) = (pmp::cotan(V.row(3) - V.row(0), V.row(11) - V.row(0)) +
                  pmp::cotan(V.row(3) - V.row(12), V.row(11) - V.row(12))) /
                 2;
    S_F(3, 12) = (pmp::cotan(V.row(3) - V.row(11), V.row(12) - V.row(11)) +
                 pmp::cotan(V.row(3) - V.row(4), V.row(12) - V.row(4))) /
                2;

    S_F(4, 5) = pmp::cotan(V.row(4) - V.row(13), V.row(5) - V.row(13)) / 2;
    S_F(4, 12) = (pmp::cotan(V.row(4) - V.row(3), V.row(12) - V.row(3)) +
                  pmp::cotan(V.row(4) - V.row(13), V.row(12) - V.row(13))) /
                 2;
    S_F(4, 13) = (pmp::cotan(V.row(4) - V.row(12), V.row(13) - V.row(12)) +
                  pmp::cotan(V.row(4) - V.row(5), V.row(13) - V.row(5))) /
                 2;

    S_F(5, 6) = (pmp::cotan(V.row(5) - V.row(13), V.row(6) - V.row(13)) +
                  pmp::cotan(V.row(5) - V.row(1), V.row(6) - V.row(1))) /
                 2;
    S_F(5, 13) = (pmp::cotan(V.row(5) - V.row(4), V.row(13) - V.row(4)) +
                  pmp::cotan(V.row(5) - V.row(6), V.row(13) - V.row(6))) /
                 2;

    S_F(6, 7) = pmp::cotan(V.row(6) - V.row(13), V.row(7) - V.row(13)) / 2;
    S_F(6, 13) = (pmp::cotan(V.row(6) - V.row(5), V.row(13) - V.row(5)) +
                  pmp::cotan(V.row(6) - V.row(7), V.row(13) - V.row(7))) /
                 2;

    S_F(7, 8) = pmp::cotan(V.row(7) - V.row(14), V.row(8) - V.row(14)) / 2;
    S_F(7, 13) = (pmp::cotan(V.row(7) - V.row(6), V.row(13) - V.row(6)) +
                  pmp::cotan(V.row(7) - V.row(14), V.row(13) - V.row(14))) /
                 2;
    S_F(7, 14) = (pmp::cotan(V.row(7) - V.row(13), V.row(14) - V.row(13)) +
                  pmp::cotan(V.row(7) - V.row(8), V.row(14) - V.row(8))) /
                 2;

    S_F(8, 9) = (pmp::cotan(V.row(8) - V.row(2), V.row(9) - V.row(2)) +
                  pmp::cotan(V.row(8) - V.row(14), V.row(9) - V.row(14))) /
                 2;
    S_F(8, 14) = (pmp::cotan(V.row(8) - V.row(9), V.row(14) - V.row(9)) +
                  pmp::cotan(V.row(8) - V.row(7), V.row(14) - V.row(7))) /
                 2;

    S_F(9, 10) = pmp::cotan(V.row(9) - V.row(14), V.row(10) - V.row(14)) / 2;
    S_F(9, 14) = (pmp::cotan(V.row(9) - V.row(8), V.row(14) - V.row(8)) +
                  pmp::cotan(V.row(9) - V.row(10), V.row(14) - V.row(10))) /
                 2;

    S_F(10, 11) = pmp::cotan(V.row(10) - V.row(12), V.row(11) - V.row(12)) / 2;
    S_F(10, 12) = (pmp::cotan(V.row(10) - V.row(11), V.row(12) - V.row(11)) +
                  pmp::cotan(V.row(10) - V.row(14), V.row(12) - V.row(14))) /
                 2;
    S_F(10, 14) = (pmp::cotan(V.row(10) - V.row(12), V.row(14) - V.row(12)) +
                  pmp::cotan(V.row(10) - V.row(9), V.row(14) - V.row(9))) /
                 2;

    S_F(11, 12) = (pmp::cotan(V.row(11) - V.row(3), V.row(12) - V.row(3)) +
                   pmp::cotan(V.row(11) - V.row(10), V.row(12) - V.row(10))) /
                  2;

    S_F(12, 13) = (pmp::cotan(V.row(12) - V.row(4), V.row(13) - V.row(4)) +
                   pmp::cotan(V.row(12) - V.row(14), V.row(13) - V.row(14))) /
                  2;
    S_F(12, 14) = (pmp::cotan(V.row(12) - V.row(10), V.row(14) - V.row(10)) +
                   pmp::cotan(V.row(12) - V.row(13), V.row(14) - V.row(13))) /
                  2;

    S_F(13, 14) = (pmp::cotan(V.row(13) - V.row(7), V.row(14) - V.row(7)) +
                   pmp::cotan(V.row(13) - V.row(12), V.row(14) - V.row(12))) /
                  2;

    for (int j = 0; j < 15; j++) //transpose
    {
        for (int i = j + 1; i < 15; i++)
        {
            S_F(i, j) = S_F(j, i);
        }
    }

    for (int i = 0; i < 15; i++)
    {
        double w = 0;
        for (int j = 0; j < 15; j++)
        {
            w += S_F(i, j);
        }
        S_F(i, i) = -w;
    }

    // sandwiching with (local) restriction and prolongation matrix
    S = -(P.transpose() * S_F * P);
    //std::cout << S_F << std::endl;
}

//----------------------------------------------------------------------------------

void setup_triangle3_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                      const std::vector<Normal> &normals,
                                      Eigen::MatrixXd &S)
{
    const int n = (int)polygon.rows(); // n == 3
    if (n != 3)
        return;
    S.resize(3, 3);

    Eigen::MatrixXd S_F, P, V;
    S_F.resize(45, 45);
    S_F.setZero();
    P.resize(45, 3);
    V.resize(45, 3);

    //construct the prolongation matrix;
    //P.row(1) << 1., 2., 3.;
    P.row(0) = Eigen::Vector3d(1., 0., 0.);
    P.row(1) = Eigen::Vector3d(0., 1., 0.);
    P.row(2) = Eigen::Vector3d(0., 0., 1.);

    P.row(3) = Eigen::Vector3d(7. / 8., 1. / 8., 0.);
    P.row(4) = Eigen::Vector3d(0.75, 0.25, 0.);
    P.row(5) = Eigen::Vector3d(5. / 8., 3. / 8., 0.);
    P.row(6) = Eigen::Vector3d(0.5, 0.5, 0.);//axis
    P.row(7) = Eigen::Vector3d(3. / 8., 5. / 8., 0.);
    P.row(8) = Eigen::Vector3d(0.25, 0.75, 0.);
    P.row(9) = Eigen::Vector3d(1. / 8., 7. / 8., 0.);

    P.row(10) = Eigen::Vector3d(0., 7. / 8., 1. / 8.);
    P.row(11) = Eigen::Vector3d(0., 0.75, 0.25);
    P.row(12) = Eigen::Vector3d(0., 5. / 8., 3. / 8.);
    P.row(13) = Eigen::Vector3d(0., 0.5, 0.5); //axis
    P.row(14) = Eigen::Vector3d(0., 3. / 8., 5. / 8.);
    P.row(15) = Eigen::Vector3d(0., 0.25, 0.75);
    P.row(16) = Eigen::Vector3d(0., 1. / 8., 7. / 8.);

    P.row(17) = Eigen::Vector3d(1. / 8., 0., 7. / 8.);
    P.row(18) = Eigen::Vector3d(0.25, 0., 0.75);
    P.row(19) = Eigen::Vector3d(3. / 8., 0., 5. / 8.);
    P.row(20) = Eigen::Vector3d(0.5, 0., 0.5); //axis
    P.row(21) = Eigen::Vector3d(5. / 8., 0., 3. / 8.);
    P.row(22) = Eigen::Vector3d(0.75, 0., 0.25);
    P.row(23) = Eigen::Vector3d(7. / 8., 0., 1. / 8.);
    
    P.row(24) = Eigen::Vector3d(0.75, 1. / 8., 1. / 8.);
    P.row(25) = Eigen::Vector3d(5. / 8., 0.25, 1. / 8.);
    P.row(26) = Eigen::Vector3d(0.5, 3. / 8., 1. / 8.);
    P.row(27) = Eigen::Vector3d(3. / 8., 0.5, 1. / 8.); 
    P.row(28) = Eigen::Vector3d(0.25, 5. / 8., 1. / 8.);
    P.row(29) = Eigen::Vector3d(1. / 8., 0.75, 1. / 8.);
    P.row(30) = Eigen::Vector3d(1. / 8., 5. / 8., 0.25);
    
    P.row(31) = Eigen::Vector3d(1. / 8., 0.5, 3. / 8.);
    P.row(32) = Eigen::Vector3d(1. / 8., 3. / 8., 0.5);
    P.row(33) = Eigen::Vector3d(1. / 8., 0.25, 5. / 8.);
    P.row(34) = Eigen::Vector3d(1. / 8., 1. / 8., 0.75);
    P.row(35) = Eigen::Vector3d(0.25, 1. / 8., 5. / 8.);
    P.row(36) = Eigen::Vector3d(3. / 8., 1. / 8., 0.5);
    P.row(37) = Eigen::Vector3d(0.5, 1. / 8., 3. / 8.);

    P.row(38) = Eigen::Vector3d(5. / 8., 1. / 8., 0.25);
    P.row(39) = Eigen::Vector3d(0.5, 0.25, 0.25);
    P.row(40) = Eigen::Vector3d(3. / 8., 3. / 8., 0.25);
    P.row(41) = Eigen::Vector3d(0.25, 0.5, 0.25);
    P.row(42) = Eigen::Vector3d(0.25, 3. / 8., 3. / 8.);
    P.row(43) = Eigen::Vector3d(0.25, 0.25, 0.5);
    P.row(44) = Eigen::Vector3d(3. / 8., 0.25, 3. / 8.);

    //construct the vertex matrix;
    //V.row(0) = polygon.row(0);
    //V.row(1) = polygon.row(1);
    //V.row(2) = polygon.row(2);
    
    // compute position of virtual vertex
    // phong tessellation
    Eigen::Vector3d vvertex;
    for (int i = 3; i < 45; i++)
    {
        vvertex = polygon.transpose() * Eigen::Vector3d(P.row(i));
        get_vvertex_hat(polygon, normals, Eigen::Vector3d(P.row(i)), vvertex);
        V.row(i) = vvertex;
    }

    //construct the vertex matrix;
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    //construct refine stiffness matrix;
    // 108 edges
    S_F(0, 3) = pmp::cotan(V.row(3) - V.row(23), V.row(0) - V.row(23)) / 2;
    S_F(0, 23) = pmp::cotan(V.row(23) - V.row(3), V.row(0) - V.row(3)) / 2;

    S_F(1, 9) = pmp::cotan(V.row(1) - V.row(10), V.row(9) - V.row(10)) / 2;
    S_F(1, 10) = pmp::cotan(V.row(1) - V.row(9), V.row(10) - V.row(9)) / 2;

    S_F(2, 16) = pmp::cotan(V.row(2) - V.row(17), V.row(16) - V.row(17)) / 2;
    S_F(2, 17) = pmp::cotan(V.row(2) - V.row(16), V.row(17) - V.row(16)) / 2;

    S_F(3, 4) = pmp::cotan(V.row(3) - V.row(24), V.row(4) - V.row(24)) / 2;
    S_F(3, 23) = (pmp::cotan(V.row(3) - V.row(0), V.row(23) - V.row(0)) +
                  pmp::cotan(V.row(3) - V.row(24), V.row(23) - V.row(24))) /
                 2;
    S_F(3, 24) = (pmp::cotan(V.row(3) - V.row(4), V.row(24) - V.row(4)) +
                  pmp::cotan(V.row(3) - V.row(23), V.row(24) - V.row(23))) /
                 2;

    S_F(4, 5) = pmp::cotan(V.row(4) - V.row(25), V.row(5) - V.row(25)) / 2;
    S_F(4, 24) = (pmp::cotan(V.row(4) - V.row(3), V.row(24) - V.row(3)) +
                  pmp::cotan(V.row(4) - V.row(25), V.row(24) - V.row(25))) /
                 2;
    S_F(4, 25) = (pmp::cotan(V.row(4) - V.row(5), V.row(25) - V.row(5)) +
                  pmp::cotan(V.row(4) - V.row(24), V.row(25) - V.row(24))) /
                 2;

    S_F(5, 6) = pmp::cotan(V.row(5) - V.row(26), V.row(6) - V.row(26)) / 2;
    S_F(5, 25) = (pmp::cotan(V.row(5) - V.row(26), V.row(25) - V.row(26)) +
                 pmp::cotan(V.row(5) - V.row(4), V.row(25) - V.row(4))) /
                2;
    S_F(5, 26) = (pmp::cotan(V.row(5) - V.row(25), V.row(26) - V.row(25)) +
                  pmp::cotan(V.row(5) - V.row(6), V.row(26) - V.row(6))) /
                 2;

    S_F(6, 7) = pmp::cotan(V.row(6) - V.row(27), V.row(7) - V.row(27)) / 2;
    S_F(6, 26) = (pmp::cotan(V.row(6) - V.row(5), V.row(26) - V.row(5)) +
                  pmp::cotan(V.row(6) - V.row(27), V.row(26) - V.row(27))) /
                 2;
    S_F(6, 27) = (pmp::cotan(V.row(6) - V.row(7), V.row(27) - V.row(7)) +
                  pmp::cotan(V.row(6) - V.row(26), V.row(27) - V.row(26))) /
                 2;

    S_F(7, 8) = pmp::cotan(V.row(7) - V.row(28), V.row(8) - V.row(28)) / 2;
    S_F(7, 27) = (pmp::cotan(V.row(7) - V.row(6), V.row(27) - V.row(6)) +
                  pmp::cotan(V.row(7) - V.row(28), V.row(27) - V.row(28))) /
                 2;
    S_F(7, 28) = (pmp::cotan(V.row(7) - V.row(27), V.row(28) - V.row(27)) +
                  pmp::cotan(V.row(7) - V.row(8), V.row(28) - V.row(8))) /
                 2;

    S_F(8, 9) = pmp::cotan(V.row(8) - V.row(29), V.row(9) - V.row(29)) / 2;
    S_F(8, 28) = (pmp::cotan(V.row(8) - V.row(7), V.row(28) - V.row(7)) +
                  pmp::cotan(V.row(8) - V.row(29), V.row(28) - V.row(29))) /
                 2;
    S_F(8, 29) = (pmp::cotan(V.row(8) - V.row(9), V.row(29) - V.row(9)) +
                  pmp::cotan(V.row(8) - V.row(28), V.row(29) - V.row(28))) /
                 2;

    S_F(9, 10) = (pmp::cotan(V.row(9) - V.row(1), V.row(10) - V.row(1)) +
                 pmp::cotan(V.row(9) - V.row(29), V.row(10) - V.row(29))) /
                2;
    S_F(9, 29) = (pmp::cotan(V.row(9) - V.row(8), V.row(29) - V.row(8)) +
                  pmp::cotan(V.row(9) - V.row(10), V.row(29) - V.row(10))) /
                 2;

    S_F(10, 11) = pmp::cotan(V.row(10) - V.row(29), V.row(11) - V.row(29)) / 2;
    S_F(10, 29) = (pmp::cotan(V.row(10) - V.row(9), V.row(29) - V.row(9)) +
                   pmp::cotan(V.row(10) - V.row(11), V.row(29) - V.row(11))) /
                  2;

    S_F(11, 12) = pmp::cotan(V.row(11) - V.row(30), V.row(12) - V.row(30)) / 2;
    S_F(11, 29) = (pmp::cotan(V.row(11) - V.row(10), V.row(29) - V.row(10)) +
                   pmp::cotan(V.row(11) - V.row(30), V.row(29) - V.row(30))) /
                  2;
    S_F(11, 30) = (pmp::cotan(V.row(11) - V.row(29), V.row(30) - V.row(29)) +
                   pmp::cotan(V.row(11) - V.row(12), V.row(30) - V.row(12))) /
                  2;

    S_F(12, 13) = pmp::cotan(V.row(12) - V.row(31), V.row(13) - V.row(31)) / 2;
    S_F(12, 30) = (pmp::cotan(V.row(12) - V.row(11), V.row(30) - V.row(11)) +
                   pmp::cotan(V.row(12) - V.row(31), V.row(30) - V.row(31))) /
                  2;
    S_F(12, 31) = (pmp::cotan(V.row(12) - V.row(30), V.row(31) - V.row(30)) +
                   pmp::cotan(V.row(12) - V.row(13), V.row(31) - V.row(13))) /
                  2;

    S_F(13, 14) = pmp::cotan(V.row(13) - V.row(32), V.row(14) - V.row(32)) / 2;
    S_F(13, 31) = (pmp::cotan(V.row(13) - V.row(12), V.row(31) - V.row(12)) +
                   pmp::cotan(V.row(13) - V.row(32), V.row(31) - V.row(32))) /
                  2;
    S_F(13, 32) = (pmp::cotan(V.row(13) - V.row(31), V.row(32) - V.row(31)) +
                   pmp::cotan(V.row(13) - V.row(14), V.row(32) - V.row(14))) /
                  2;

    S_F(14, 15) = pmp::cotan(V.row(14) - V.row(33), V.row(15) - V.row(33)) / 2;
    S_F(14, 32) = (pmp::cotan(V.row(14) - V.row(13), V.row(32) - V.row(13)) +
                   pmp::cotan(V.row(14) - V.row(33), V.row(32) - V.row(33))) /
                  2;
    S_F(14, 33) = (pmp::cotan(V.row(14) - V.row(32), V.row(33) - V.row(32)) +
                   pmp::cotan(V.row(14) - V.row(15), V.row(33) - V.row(15))) /
                  2;

    S_F(15, 16) = pmp::cotan(V.row(15) - V.row(34), V.row(16) - V.row(34)) / 2;
    S_F(15, 33) = (pmp::cotan(V.row(15) - V.row(14), V.row(33) - V.row(14)) +
                   pmp::cotan(V.row(15) - V.row(34), V.row(33) - V.row(34))) /
                  2;
    S_F(15, 34) = (pmp::cotan(V.row(15) - V.row(33), V.row(34) - V.row(33)) +
                   pmp::cotan(V.row(15) - V.row(16), V.row(34) - V.row(16))) /
                  2;

    S_F(16, 17) = (pmp::cotan(V.row(16) - V.row(2), V.row(17) - V.row(2)) +
                   pmp::cotan(V.row(16) - V.row(34), V.row(17) - V.row(34))) /
                  2;
    S_F(16, 34) = (pmp::cotan(V.row(16) - V.row(15), V.row(34) - V.row(15)) +
                   pmp::cotan(V.row(16) - V.row(17), V.row(34) - V.row(17))) /
                  2;

    S_F(17, 18) = pmp::cotan(V.row(17) - V.row(34), V.row(18) - V.row(34)) / 2;
    S_F(17, 34) = (pmp::cotan(V.row(17) - V.row(18), V.row(34) - V.row(18)) +
                   pmp::cotan(V.row(17) - V.row(16), V.row(34) - V.row(16))) /
                  2;

    S_F(18, 19) = pmp::cotan(V.row(18) - V.row(35), V.row(19) - V.row(35)) / 2;
    S_F(18, 34) = (pmp::cotan(V.row(18) - V.row(17), V.row(34) - V.row(17)) +
                   pmp::cotan(V.row(18) - V.row(35), V.row(34) - V.row(35))) /
                  2;
    S_F(18, 35) = (pmp::cotan(V.row(18) - V.row(34), V.row(35) - V.row(34)) +
                   pmp::cotan(V.row(18) - V.row(19), V.row(35) - V.row(19))) /
                  2;

    S_F(19, 20) = pmp::cotan(V.row(19) - V.row(36), V.row(20) - V.row(36)) / 2;
    S_F(19, 35) = (pmp::cotan(V.row(19) - V.row(18), V.row(35) - V.row(18)) +
                   pmp::cotan(V.row(19) - V.row(36), V.row(35) - V.row(36))) /
                  2;
    S_F(19, 36) = (pmp::cotan(V.row(19) - V.row(35), V.row(36) - V.row(35)) +
                   pmp::cotan(V.row(19) - V.row(20), V.row(36) - V.row(20))) /
                  2;

    S_F(20, 21) = pmp::cotan(V.row(20) - V.row(37), V.row(21) - V.row(37)) / 2;
    S_F(20, 36) = (pmp::cotan(V.row(20) - V.row(19), V.row(36) - V.row(19)) +
                   pmp::cotan(V.row(20) - V.row(37), V.row(36) - V.row(37))) /
                  2;
    S_F(20, 37) = (pmp::cotan(V.row(20) - V.row(36), V.row(37) - V.row(36)) +
                   pmp::cotan(V.row(20) - V.row(21), V.row(37) - V.row(21))) /
                  2;

    S_F(21, 22) = pmp::cotan(V.row(21) - V.row(38), V.row(22) - V.row(38)) / 2;
    S_F(21, 37) = (pmp::cotan(V.row(21) - V.row(20), V.row(37) - V.row(20)) +
                   pmp::cotan(V.row(21) - V.row(38), V.row(37) - V.row(38))) /
                  2;
    S_F(21, 38) = (pmp::cotan(V.row(21) - V.row(37), V.row(38) - V.row(37)) +
                   pmp::cotan(V.row(21) - V.row(22), V.row(38) - V.row(22))) /
                  2;

    S_F(22, 23) = pmp::cotan(V.row(22) - V.row(24), V.row(23) - V.row(24)) / 2;
    S_F(22, 24) = (pmp::cotan(V.row(22) - V.row(23), V.row(24) - V.row(23)) +
                   pmp::cotan(V.row(22) - V.row(38), V.row(24) - V.row(38))) /
                  2;
    S_F(22, 38) = (pmp::cotan(V.row(22) - V.row(21), V.row(38) - V.row(21)) +
                   pmp::cotan(V.row(22) - V.row(24), V.row(38) - V.row(24))) /
                  2;

    S_F(23, 24) = (pmp::cotan(V.row(23) - V.row(3), V.row(24) - V.row(3)) +
                   pmp::cotan(V.row(23) - V.row(22), V.row(24) - V.row(22))) /
                  2;

    S_F(24, 25) = (pmp::cotan(V.row(24) - V.row(4), V.row(25) - V.row(4)) +
                   pmp::cotan(V.row(24) - V.row(38), V.row(25) - V.row(38))) /
                  2;
    S_F(24, 38) = (pmp::cotan(V.row(24) - V.row(22), V.row(38) - V.row(22)) +
                   pmp::cotan(V.row(24) - V.row(25), V.row(38) - V.row(25))) /
                  2;

    S_F(25, 26) = (pmp::cotan(V.row(25) - V.row(5), V.row(26) - V.row(5)) +
                   pmp::cotan(V.row(25) - V.row(39), V.row(26) - V.row(39))) /
                  2;
    S_F(25, 38) = (pmp::cotan(V.row(25) - V.row(24), V.row(38) - V.row(24)) +
                   pmp::cotan(V.row(25) - V.row(39), V.row(38) - V.row(39))) /
                  2;
    S_F(25, 39) = (pmp::cotan(V.row(25) - V.row(26), V.row(39) - V.row(26)) +
                   pmp::cotan(V.row(25) - V.row(38), V.row(39) - V.row(38))) /
                  2;

    S_F(26, 27) = (pmp::cotan(V.row(26) - V.row(6), V.row(27) - V.row(6)) +
                   pmp::cotan(V.row(26) - V.row(40), V.row(27) - V.row(40))) /
                  2;
    S_F(26, 39) = (pmp::cotan(V.row(26) - V.row(25), V.row(39) - V.row(25)) +
                   pmp::cotan(V.row(26) - V.row(40), V.row(39) - V.row(40))) /
                  2;
    S_F(26, 40) = (pmp::cotan(V.row(26) - V.row(27), V.row(40) - V.row(27)) +
                   pmp::cotan(V.row(26) - V.row(39), V.row(40) - V.row(39))) /
                  2;

    S_F(27, 28) = (pmp::cotan(V.row(27) - V.row(7), V.row(28) - V.row(7)) +
                   pmp::cotan(V.row(27) - V.row(41), V.row(28) - V.row(41))) /
                  2;
    S_F(27, 40) = (pmp::cotan(V.row(27) - V.row(26), V.row(40) - V.row(26)) +
                   pmp::cotan(V.row(27) - V.row(41), V.row(40) - V.row(41))) /
                  2;
    S_F(27, 41) = (pmp::cotan(V.row(27) - V.row(28), V.row(41) - V.row(28)) +
                   pmp::cotan(V.row(27) - V.row(40), V.row(41) - V.row(40))) /
                  2;

    S_F(28, 29) = (pmp::cotan(V.row(28) - V.row(8), V.row(29) - V.row(8)) +
                   pmp::cotan(V.row(28) - V.row(30), V.row(29) - V.row(30))) /
                  2;
    S_F(28, 30) = (pmp::cotan(V.row(28) - V.row(29), V.row(30) - V.row(29)) +
                   pmp::cotan(V.row(28) - V.row(41), V.row(30) - V.row(41))) /
                  2;
    S_F(28, 41) = (pmp::cotan(V.row(28) - V.row(27), V.row(41) - V.row(27)) +
                   pmp::cotan(V.row(28) - V.row(30), V.row(41) - V.row(30))) /
                  2;

    S_F(29, 30) = (pmp::cotan(V.row(29) - V.row(11), V.row(30) - V.row(11)) +
                   pmp::cotan(V.row(29) - V.row(28), V.row(30) - V.row(28))) /
                  2;

    S_F(30, 31) = (pmp::cotan(V.row(30) - V.row(12), V.row(31) - V.row(12)) +
                   pmp::cotan(V.row(30) - V.row(41), V.row(31) - V.row(41))) /
                  2;
    S_F(30, 41) = (pmp::cotan(V.row(30) - V.row(28), V.row(41) - V.row(28)) +
                   pmp::cotan(V.row(30) - V.row(31), V.row(41) - V.row(31))) /
                  2;

    S_F(31, 32) = (pmp::cotan(V.row(31) - V.row(13), V.row(32) - V.row(13)) +
                   pmp::cotan(V.row(31) - V.row(42), V.row(32) - V.row(42))) /
                  2;
    S_F(31, 41) = (pmp::cotan(V.row(31) - V.row(30), V.row(41) - V.row(30)) +
                   pmp::cotan(V.row(31) - V.row(42), V.row(41) - V.row(42))) /
                  2;
    S_F(31, 42) = (pmp::cotan(V.row(31) - V.row(32), V.row(42) - V.row(32)) +
                   pmp::cotan(V.row(31) - V.row(41), V.row(42) - V.row(41))) /
                  2;

    S_F(32, 33) = (pmp::cotan(V.row(32) - V.row(14), V.row(33) - V.row(14)) +
                   pmp::cotan(V.row(32) - V.row(43), V.row(33) - V.row(43))) /
                  2;
    S_F(32, 42) = (pmp::cotan(V.row(32) - V.row(31), V.row(42) - V.row(31)) +
                   pmp::cotan(V.row(32) - V.row(43), V.row(42) - V.row(43))) /
                  2;
    S_F(32, 43) = (pmp::cotan(V.row(32) - V.row(33), V.row(43) - V.row(33)) +
                   pmp::cotan(V.row(32) - V.row(42), V.row(43) - V.row(42))) /
                  2;

    S_F(33, 34) = (pmp::cotan(V.row(33) - V.row(15), V.row(34) - V.row(15)) +
                   pmp::cotan(V.row(33) - V.row(35), V.row(34) - V.row(35))) /
                  2;
    S_F(33, 35) = (pmp::cotan(V.row(33) - V.row(34), V.row(35) - V.row(34)) +
                   pmp::cotan(V.row(33) - V.row(43), V.row(35) - V.row(43))) /
                  2;
    S_F(33, 43) = (pmp::cotan(V.row(33) - V.row(32), V.row(43) - V.row(32)) +
                   pmp::cotan(V.row(33) - V.row(35), V.row(43) - V.row(35))) /
                  2;

    S_F(34, 35) = (pmp::cotan(V.row(34) - V.row(18), V.row(35) - V.row(18)) +
                   pmp::cotan(V.row(34) - V.row(33), V.row(35) - V.row(33))) /
                  2;

    S_F(35, 36) = (pmp::cotan(V.row(35) - V.row(19), V.row(36) - V.row(19)) +
                   pmp::cotan(V.row(35) - V.row(43), V.row(36) - V.row(43))) /
                  2;
    S_F(35, 43) = (pmp::cotan(V.row(35) - V.row(33), V.row(43) - V.row(33)) +
                   pmp::cotan(V.row(35) - V.row(36), V.row(43) - V.row(36))) /
                  2;

    S_F(36, 37) = (pmp::cotan(V.row(36) - V.row(20), V.row(37) - V.row(20)) +
                   pmp::cotan(V.row(36) - V.row(44), V.row(37) - V.row(44))) /
                  2;
    S_F(36, 43) = (pmp::cotan(V.row(36) - V.row(35), V.row(43) - V.row(35)) +
                   pmp::cotan(V.row(36) - V.row(44), V.row(43) - V.row(44))) /
                  2;
    S_F(36, 44) = (pmp::cotan(V.row(36) - V.row(37), V.row(44) - V.row(37)) +
                   pmp::cotan(V.row(36) - V.row(43), V.row(44) - V.row(43))) /
                  2;

    S_F(37, 38) = (pmp::cotan(V.row(37) - V.row(21), V.row(38) - V.row(21)) +
                   pmp::cotan(V.row(37) - V.row(39), V.row(38) - V.row(39))) /
                  2;
    S_F(37, 39) = (pmp::cotan(V.row(37) - V.row(38), V.row(39) - V.row(38)) +
                   pmp::cotan(V.row(37) - V.row(44), V.row(39) - V.row(44))) /
                  2;
    S_F(37, 44) = (pmp::cotan(V.row(37) - V.row(36), V.row(44) - V.row(36)) +
                   pmp::cotan(V.row(37) - V.row(39), V.row(44) - V.row(39))) /
                  2;

    S_F(38, 39) = (pmp::cotan(V.row(38) - V.row(25), V.row(39) - V.row(25)) +
                   pmp::cotan(V.row(38) - V.row(37), V.row(39) - V.row(37))) /
                  2;

    S_F(39, 40) = (pmp::cotan(V.row(39) - V.row(26), V.row(40) - V.row(26)) +
                   pmp::cotan(V.row(39) - V.row(44), V.row(40) - V.row(44))) /
                  2;
    S_F(39, 44) = (pmp::cotan(V.row(39) - V.row(37), V.row(44) - V.row(37)) +
                   pmp::cotan(V.row(39) - V.row(40), V.row(44) - V.row(40))) /
                  2;

    S_F(40, 41) = (pmp::cotan(V.row(40) - V.row(27), V.row(41) - V.row(27)) +
                   pmp::cotan(V.row(40) - V.row(42), V.row(41) - V.row(42))) /
                  2;
    S_F(40, 42) = (pmp::cotan(V.row(40) - V.row(41), V.row(42) - V.row(41)) +
                   pmp::cotan(V.row(40) - V.row(44), V.row(42) - V.row(44))) /
                  2;
    S_F(40, 44) = (pmp::cotan(V.row(40) - V.row(39), V.row(44) - V.row(39)) +
                   pmp::cotan(V.row(40) - V.row(42), V.row(44) - V.row(42))) /
                  2;

    S_F(41, 42) = (pmp::cotan(V.row(41) - V.row(31), V.row(42) - V.row(31)) +
                   pmp::cotan(V.row(41) - V.row(40), V.row(42) - V.row(40))) /
                  2;

    S_F(42, 43) = (pmp::cotan(V.row(42) - V.row(32), V.row(43) - V.row(32)) +
                   pmp::cotan(V.row(42) - V.row(44), V.row(43) - V.row(44))) /
                  2;
    S_F(42, 44) = (pmp::cotan(V.row(42) - V.row(40), V.row(44) - V.row(40)) +
                   pmp::cotan(V.row(42) - V.row(43), V.row(44) - V.row(43))) /
                  2;

    S_F(43, 44) = (pmp::cotan(V.row(43) - V.row(36), V.row(44) - V.row(36)) +
                   pmp::cotan(V.row(43) - V.row(42), V.row(44) - V.row(42))) /
                  2;

    for (int j = 0; j < 45; j++) //transpose
    {
        for (int i = j + 1; i < 45; i++)
        {
            S_F(i, j) = S_F(j, i);
        }
    }

    for (int i = 0; i < 45; i++)
    {
        double w = 0.;
        for (int j = 0; j < 45; j++)
        {
            w += S_F(i, j);
        }
        S_F(i, i) = -w;
    }

    // sandwiching with (local) restriction and prolongation matrix
    S = -(P.transpose() * S_F * P);
    //std::cout << S_F << std::endl;
}

//----------------------------------------------------------------------------------

void setup_quadratic_stiffness_matrix(const Eigen::MatrixXd &polygon,
                                      const std::vector<Normal> &normals,
                                      Eigen::MatrixXd &S)
{
    const int n = (int)polygon.rows(); // n == 3
    if (n != 3)
        return;
    S.resize(3, 3);

    Eigen::MatrixXd S_F, P, V, P0;
    S_F.resize(9, 9);
    S_F.setZero();
    P.resize(9, 6);
    P0.resize(6, 3);
    V.resize(9, 3);

    //construct the prolongation matrix;
    P << 1., 0., 0., 0., 0., 0.,
         0., 1., 0., 0., 0., 0., 
         0., 0., 1., 0., 0., 0.,
         0., 0., 0., 1., 0., 0.,
         0., 0., 0., 0., 1., 0.,
         0., 0., 0., 0., 0., 1.,
         //0.5, 0.25, 0.25, 0., 0., 0., 
         //0.5, 0.5, 0.25, 0., 0., 0., 
         //0.25, 0.25, 0.5, 0., 0., 0.;
        0., 0., 0., 0.5, 0.5, 0.,
        0., 0., 0., 0., 0.5, 0.5,
        0., 0., 0., 0.5, 0., 0.5;

    //construct the vertex matrix，2*n vertex
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    V.row(3) = 0.5 * (V.row(0) + V.row(1));
    V.row(4) = 0.5 * (V.row(1) + V.row(2));
    V.row(5) = 0.5 * (V.row(0) + V.row(2));
    //n virtual vertex;
    V.row(6) = 0.5 * (V.row(3) + V.row(5));
    V.row(7) = 0.5 * (V.row(3) + V.row(4));
    V.row(8) = 0.5 * (V.row(4) + V.row(5));

    //construct refine stiffness matrix;
    S_F(0, 3) = pmp::cotan(V.row(0) - V.row(6), V.row(3) - V.row(6)) / 2;
    S_F(0, 5) = pmp::cotan(V.row(0) - V.row(6), V.row(5) - V.row(6)) / 2;
    S_F(0, 6) = (pmp::cotan(V.row(0) - V.row(3), V.row(6) - V.row(3)) +
                 pmp::cotan(V.row(0) - V.row(5), V.row(6) - V.row(5))) /
                2;

    S_F(1, 3) = pmp::cotan(V.row(3) - V.row(7), V.row(1) - V.row(7)) / 2;
    S_F(1, 4) = pmp::cotan(V.row(1) - V.row(7), V.row(4) - V.row(7)) / 2;
    S_F(1, 7) = (pmp::cotan(V.row(1) - V.row(3), V.row(7) - V.row(3)) +
                 pmp::cotan(V.row(1) - V.row(4), V.row(7) - V.row(4))) /
                2;

    S_F(2, 4) = pmp::cotan(V.row(2) - V.row(8), V.row(4) - V.row(8)) / 2;
    S_F(2, 5) = pmp::cotan(V.row(2) - V.row(8), V.row(5) - V.row(8)) / 2;
    S_F(2, 8) = (pmp::cotan(V.row(2) - V.row(4), V.row(8) - V.row(4)) +
                 pmp::cotan(V.row(2) - V.row(5), V.row(8) - V.row(5))) /
                2;

    S_F(3, 6) = (pmp::cotan(V.row(3) - V.row(0), V.row(6) - V.row(0)) +
                 pmp::cotan(V.row(3) - V.row(7), V.row(6) - V.row(7))) /
                2;
    S_F(3, 7) = (pmp::cotan(V.row(3) - V.row(1), V.row(7) - V.row(1)) +
                 pmp::cotan(V.row(3) - V.row(6), V.row(7) - V.row(6))) /
                2;

    S_F(4, 7) = (pmp::cotan(V.row(4) - V.row(1), V.row(7) - V.row(1)) +
                 pmp::cotan(V.row(4) - V.row(8), V.row(7) - V.row(8))) /
                2;
    S_F(4, 8) = (pmp::cotan(V.row(4) - V.row(2), V.row(8) - V.row(2)) +
                 pmp::cotan(V.row(4) - V.row(7), V.row(8) - V.row(7))) /
                2;

    S_F(5, 6) = (pmp::cotan(V.row(5) - V.row(0), V.row(6) - V.row(0)) +
                 pmp::cotan(V.row(5) - V.row(8), V.row(6) - V.row(8))) /
                2;
    S_F(5, 8) = (pmp::cotan(V.row(5) - V.row(2), V.row(8) - V.row(2)) +
                 pmp::cotan(V.row(5) - V.row(6), V.row(8) - V.row(6))) /
                2;

    S_F(6, 7) = (pmp::cotan(V.row(6) - V.row(3), V.row(7) - V.row(3)) +
                 pmp::cotan(V.row(6) - V.row(8), V.row(7) - V.row(8))) /
                2;
    S_F(6, 8) = (pmp::cotan(V.row(6) - V.row(5), V.row(8) - V.row(5)) +
                 pmp::cotan(V.row(6) - V.row(7), V.row(8) - V.row(7))) /
                2;

    S_F(7, 8) = (pmp::cotan(V.row(7) - V.row(4), V.row(8) - V.row(4)) +
                 pmp::cotan(V.row(7) - V.row(6), V.row(8) - V.row(6))) /
                2;

    for (int j = 0; j < 9; j++) //transpose
    {
        for (int i = j + 1; i < 9; i++)
        {
            S_F(i, j) = S_F(j, i);
        }
    }

    for (int i = 0; i < 9; i++)
    {
        double w = 0;
        for (int j = 0; j < 9; j++)
        {
            w += S_F(i, j);
        }
        S_F(i, i) = -w;
    }

    P0 << 1., 0., 0.,
         0., 1., 0.,
         0., 0., 1.,
         0.5, 0.5, 0., 
         0., 0.5, 0.5,
         0.5, 0., 0.5;

    // sandwiching with (local) restriction and prolongation matrix
    S = - (P0.transpose() * (P.transpose() * S_F * P) * P0);
    //std::cout << S_F << std::endl;
}

//----------------------------------------------------------------------------------

void setup_mass_matrix(const SurfaceMesh &mesh, Eigen::SparseMatrix<double> &M, int flag)
{
    const int nv = mesh.n_vertices();

    std::vector<Vertex> vertices; // polygon vertices
    Eigen::MatrixXd polygon;      // positions of polygon vertices
    std::vector<Normal> normals;  // polygon normals
    Eigen::VectorXd weights;      // affine weights of virtual vertex
    Eigen::MatrixXd Mi;           // local mass matrix

    std::vector<Eigen::Triplet<double>> trip;

    for (Face f : mesh.faces())
    {
        // collect polygon vertices
        vertices.clear();
        for (Vertex v : mesh.vertices(f))
        {
            vertices.push_back(v);
        }
        const int n=vertices.size();

        // collect their positions
        polygon.resize(n, 3);
        for (int i=0; i<n; ++i)
        {
            polygon.row(i) = (Eigen::Vector3d) mesh.position(vertices[i]);
            
            pre_compute_normals(mesh);
            normals.push_back(pre_normals[vertices[i].idx()]);
            //normals.push_back(pmp::SurfaceNormals::compute_vertex_normal(mesh, vertices[i]));
        }

        // compute virtual vertex, setup local mass matrix
        if (flag == 0)
        {
            compute_virtual_vertex(polygon, weights);
            setup_polygon_mass_matrix(polygon, normals, weights, Mi);
        }
        else if (flag == 1)
            setup_triangle_mass_matrix(polygon, normals, Mi); //subdivision once
        else if (flag == 2)
            setup_triangle2_mass_matrix(polygon, normals, Mi); //subdivison twice
        else if (flag == 3)
            setup_triangle3_mass_matrix(polygon, normals, Mi); //subdivison three times
        else 
            setup_quadratic_mass_matrix(polygon, normals, Mi); //quadratic

        // assemble into global mass matrix
        for (int j=0; j<n; ++j)
        {
            for (int k=0; k<n; ++k)
            {
                trip.emplace_back(vertices[k].idx(), vertices[j].idx(), Mi(k, j));
            }
        }
    }

    // build sparse matrix from triplets
    M.resize(nv, nv);
    M.setFromTriplets(trip.begin(), trip.end());

    // optional: lump mass matrix
    if (lump_mass_matrix_)
    {
        lump_matrix(M);
    }
}

//----------------------------------------------------------------------------------

void setup_polygon_mass_matrix(const Eigen::MatrixXd &polygon,
                               const std::vector<Normal> &normals,
                               const Eigen::VectorXd &vweights,
                               Eigen::MatrixXd &M)
{
    const int n = (int)polygon.rows();
    M.resize(n, n);
    M.setZero();

    // compute position of virtual vertex
    Eigen::Vector3d vvertex = polygon.transpose() * vweights;
    //phong tessellation 
    get_vvertex_hat(polygon, normals, vweights, vvertex);

    Eigen::VectorXd ln(n + 1);
    ln.setZero();

    double l[3], l2[3];

    for (int i = 0; i < n; ++i)
    {
        const int i1 = (i + 1) % n;

        l2[2] = (polygon.row(i) - polygon.row(i1)).squaredNorm();
        l2[0] = (polygon.row(i1) - vvertex.transpose()).squaredNorm();
        l2[1] = (polygon.row(i) - vvertex.transpose()).squaredNorm();

        l[0] = sqrt(l2[0]);
        l[1] = sqrt(l2[1]);
        l[2] = sqrt(l2[2]);

        const double arg = (l[0] + (l[1] + l[2])) * (l[2] - (l[0] - l[1])) *
                           (l[2] + (l[0] - l[1])) * (l[0] + (l[1] - l[2]));
        const double area = 0.25 * sqrt(arg);

        l[0] = 1.0 / 6.0 * area;
        l[1] = 1.0 / 12.0 * area;

        M(i1, i1) += 1.0 / 6.0 * area;
        M(i, i) += 1.0 / 6.0 * area;
        M(i1, i) += 1.0 / 12.0 * area;
        M(i, i1) += 1.0 / 12.0 * area;

        ln(i1) += l[1];
        ln(i) += l[1];
        ln(n) += l[0];
    }

    // sandwiching with (local) restriction and prolongation matrix
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            M(i, j) += vweights(i) * ln(j) + vweights(j) * ln(i) + vweights(i) * vweights(j) * ln(n);
}

//-----------------------------------------------------------------------------

void setup_triangle_mass_matrix(const Eigen::MatrixXd &polygon,
                                const std::vector<Normal> &normals,
                                Eigen::MatrixXd &M)
{
    const int n = (int)polygon.rows(); // n == 3
    if (n != 3)
        return;
    M.resize(3, 3);

    Eigen::MatrixXd M_F, P, V;
    M_F.resize(6, 6);
    M_F.setZero();
    P.resize(6, 3);
    V.resize(6, 3);

    //construct the prolongation matrix;
    P.row(0) = Eigen::Vector3d(1., 0., 0.);
    P.row(1) = Eigen::Vector3d(0., 1., 0.);
    P.row(2) = Eigen::Vector3d(0., 0., 1.);
    P.row(3) = Eigen::Vector3d(0.5, 0.5, 0.);
    P.row(4) = Eigen::Vector3d(0., 0.5, 0.5);
    P.row(5) = Eigen::Vector3d(0.5, 0., 0.5);

    //construct the vertex matrix;
    //V.row(0) = polygon.row(0);
    //V.row(1) = polygon.row(1);
    //V.row(2) = polygon.row(2);

    // compute position of virtual vertex
    Eigen::Vector3d vvertex =
        polygon.transpose() * Eigen::Vector3d(0.5, 0.5, 0.);
    // phong tessellation
    get_vvertex_hat(polygon, normals, Eigen::Vector3d(0.5, 0.5, 0.), vvertex);
    V.row(3) = vvertex;

    // compute position of virtual vertex
    vvertex = polygon.transpose() * Eigen::Vector3d(0., 0.5, 0.5);
    // phong tessellation
    get_vvertex_hat(polygon, normals, Eigen::Vector3d(0., 0.5, 0.5), vvertex);
    V.row(4) = vvertex;

    // compute position of virtual vertex
    vvertex = polygon.transpose() * Eigen::Vector3d(0.5, 0., 0.5);
    // phong tessellation
    get_vvertex_hat(polygon, normals, Eigen::Vector3d(0.5, 0., 0.5), vvertex);
    V.row(5) = vvertex;

    //construct the vertex matrix;
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    //construct refine mass matrix;
    M_F(0, 3) = pmp::triangle_area(V.row(0), V.row(3), V.row(5)) / 12;
    M_F(0, 5) = pmp::triangle_area(V.row(0), V.row(3), V.row(5)) / 12;

    M_F(1, 3) = pmp::triangle_area(V.row(1), V.row(3), V.row(4)) / 12;
    M_F(1, 4) = pmp::triangle_area(V.row(1), V.row(3), V.row(4)) / 12;

    M_F(2, 4) = pmp::triangle_area(V.row(2), V.row(4), V.row(5)) / 12;
    M_F(2, 5) = pmp::triangle_area(V.row(2), V.row(4), V.row(5)) / 12;

    M_F(3, 4) = (pmp::triangle_area(V.row(1), V.row(3), V.row(4)) +
                 pmp::triangle_area(V.row(3), V.row(4), V.row(5))) / 12;

    M_F(3, 5) = (pmp::triangle_area(V.row(0), V.row(3), V.row(5)) +
                 pmp::triangle_area(V.row(3), V.row(4), V.row(5))) / 12;

    M_F(4, 5) = (pmp::triangle_area(V.row(2), V.row(4), V.row(5)) +
                 pmp::triangle_area(V.row(3), V.row(4), V.row(5))) / 12;

    for (int j = 0; j < 6; j++) //transpose
    {
        for (int i = j + 1; i < 6; i++)
        {
            M_F(i, j) = M_F(j, i);
        }
    }

    for (int i = 0; i < 6; i++)
    {
        double w = 0;
        for (int j = 0; j < 6; j++)
        {
            w += M_F(i, j);
        }
        M_F(i, i) = w;
    }

    // sandwiching with (local) restriction and prolongation matrix
    M = P.transpose() * M_F * P;
}

//-----------------------------------------------------------------------------

void setup_triangle2_mass_matrix(const Eigen::MatrixXd &polygon,
                                const std::vector<Normal> &normals,
                                Eigen::MatrixXd &M)
{
    const int n = (int)polygon.rows(); // n == 3
    if (n != 3)
        return;
    M.resize(3, 3);

    Eigen::MatrixXd M_F, P, V;
    M_F.resize(15, 15);
    M_F.setZero();
    P.resize(15, 3);
    V.resize(15, 3);

    //construct the prolongation matrix;
    P.row(0) = Eigen::Vector3d(1., 0., 0.);
    P.row(1) = Eigen::Vector3d(0., 1., 0.);
    P.row(2) = Eigen::Vector3d(0., 0., 1.);

    P.row(3) = Eigen::Vector3d(0.75, 0.25, 0.);
    P.row(4) = Eigen::Vector3d(0.5, 0.5, 0.);
    P.row(5) = Eigen::Vector3d(0.25, 0.75, 0.);

    P.row(6) = Eigen::Vector3d(0., 0.75, 0.25);
    P.row(7) = Eigen::Vector3d(0., 0.5, 0.5);
    P.row(8) = Eigen::Vector3d(0., 0.25, 0.75);

    P.row(9) = Eigen::Vector3d(0.25, 0., 0.75);
    P.row(10) = Eigen::Vector3d(0.5, 0., 0.5);
    P.row(11) = Eigen::Vector3d(0.75, 0., 0.25);

    P.row(12) = Eigen::Vector3d(0.5, 0.25, 0.25);
    P.row(13) = Eigen::Vector3d(0.25, 0.5, 0.25);
    P.row(14) = Eigen::Vector3d(0.25, 0.25, 0.5);

    //construct the vertex matrix;
    //V.row(0) = polygon.row(0);
    //V.row(1) = polygon.row(1);
    //V.row(2) = polygon.row(2);

    // compute position of virtual vertex
    // phong tessellation
    Eigen::Vector3d vvertex;
    for (int i = 3; i < 15; i++)
    {
        vvertex = polygon.transpose() * Eigen::Vector3d(P.row(i));
        get_vvertex_hat(polygon, normals, Eigen::Vector3d(P.row(i)), vvertex);
        V.row(i) = vvertex;
    }

    //construct the vertex matrix;
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    //construct refine stiffness matrix;
    // 30 edges
    M_F(0, 3) = pmp::triangle_area(V.row(0), V.row(3), V.row(11)) / 12;
    M_F(0, 11) = pmp::triangle_area(V.row(0), V.row(3), V.row(11)) / 12;

    M_F(1, 5) = pmp::triangle_area(V.row(1), V.row(5), V.row(6)) / 12;
    M_F(1, 6) = pmp::triangle_area(V.row(1), V.row(5), V.row(6)) / 12;

    M_F(2, 8) = pmp::triangle_area(V.row(2), V.row(8), V.row(9)) / 12;
    M_F(2, 9) = pmp::triangle_area(V.row(2), V.row(8), V.row(9)) / 12;

    M_F(3, 4) = pmp::triangle_area(V.row(3), V.row(4), V.row(12)) / 12;
    M_F(3, 11) = (pmp::triangle_area(V.row(0), V.row(3), V.row(11)) +
                 pmp::triangle_area(V.row(3), V.row(11), V.row(12))) /
                 12;
    M_F(3, 12) = (pmp::triangle_area(V.row(3), V.row(4), V.row(12)) +
                  pmp::triangle_area(V.row(3), V.row(11), V.row(12))) /
                 12;

    M_F(4, 5) = pmp::triangle_area(V.row(4), V.row(5), V.row(13)) / 12;
    M_F(4, 12) = (pmp::triangle_area(V.row(3), V.row(4), V.row(12)) +
                  pmp::triangle_area(V.row(4), V.row(12), V.row(13))) /
                 12;
    M_F(4, 13) = (pmp::triangle_area(V.row(4), V.row(5), V.row(13)) +
                  pmp::triangle_area(V.row(4), V.row(12), V.row(13))) /
                 12;

    M_F(5, 6) = (pmp::triangle_area(V.row(1), V.row(5), V.row(6)) +
                 pmp::triangle_area(V.row(5), V.row(6), V.row(13))) /
                12;
    M_F(5, 13) = (pmp::triangle_area(V.row(4), V.row(5), V.row(13)) +
                  pmp::triangle_area(V.row(5), V.row(6), V.row(13))) /
                 12;

    M_F(6, 7) = pmp::triangle_area(V.row(6), V.row(7), V.row(13)) / 12;
    M_F(6, 13) = (pmp::triangle_area(V.row(5), V.row(6), V.row(13)) +
                  pmp::triangle_area(V.row(6), V.row(7), V.row(13))) /
                 12;

    M_F(7, 8) = pmp::triangle_area(V.row(7), V.row(8), V.row(14)) / 12;
    M_F(7, 13) = (pmp::triangle_area(V.row(6), V.row(7), V.row(13)) +
                  pmp::triangle_area(V.row(7), V.row(13), V.row(14))) /
                 12;
    M_F(7, 14) = (pmp::triangle_area(V.row(7), V.row(13), V.row(14)) +
                  pmp::triangle_area(V.row(7), V.row(8), V.row(14))) /
                 12;

    M_F(8, 9) = (pmp::triangle_area(V.row(2), V.row(8), V.row(9)) +
                 pmp::triangle_area(V.row(8), V.row(9), V.row(14))) /
                12;
    M_F(8, 14) = (pmp::triangle_area(V.row(7), V.row(8), V.row(14)) +
                  pmp::triangle_area(V.row(8), V.row(9), V.row(14))) /
                 12;

    M_F(9, 10) = pmp::triangle_area(V.row(9), V.row(10), V.row(14)) / 12;
    M_F(9, 14) = (pmp::triangle_area(V.row(8), V.row(9), V.row(14)) +
                  pmp::triangle_area(V.row(9), V.row(10), V.row(14))) /
                 12;

    M_F(10, 11) = pmp::triangle_area(V.row(10), V.row(11), V.row(12)) / 12;
    M_F(10, 12) = (pmp::triangle_area(V.row(10), V.row(11), V.row(12)) +
                   pmp::triangle_area(V.row(10), V.row(12), V.row(14))) /
                  12;
    M_F(10, 14) = (pmp::triangle_area(V.row(9), V.row(10), V.row(14)) +
                   pmp::triangle_area(V.row(10), V.row(12), V.row(14))) /
                  12;

    M_F(11, 12) = (pmp::triangle_area(V.row(3), V.row(11), V.row(12)) +
                   pmp::triangle_area(V.row(10), V.row(11), V.row(12))) /
                  12;

    M_F(12, 13) = (pmp::triangle_area(V.row(4), V.row(12), V.row(13)) +
                   pmp::triangle_area(V.row(12), V.row(13), V.row(14))) /
                  12;
    M_F(12, 14) = (pmp::triangle_area(V.row(12), V.row(13), V.row(14)) +
                   pmp::triangle_area(V.row(10), V.row(12), V.row(14))) /
                  12;

    M_F(13, 14) = (pmp::triangle_area(V.row(12), V.row(13), V.row(14)) +
                   pmp::triangle_area(V.row(7), V.row(13), V.row(14))) /
                  12;

    for (int j = 0; j < 15; j++) //transpose
    {
        for (int i = j + 1; i < 15; i++)
        {
            M_F(i, j) = M_F(j, i);
        }
    }

    for (int i = 0; i < 15; i++)
    {
        double w = 0;
        for (int j = 0; j < 15; j++)
        {
            w += M_F(i, j);
        }
        M_F(i, i) = w;
    }

    // sandwiching with (local) restriction and prolongation matrix
    M = P.transpose() * M_F * P;
    //std::cout << M_F << std::endl;
}

//-----------------------------------------------------------------------------

void setup_triangle3_mass_matrix(const Eigen::MatrixXd &polygon,
                                 const std::vector<Normal> &normals,
                                 Eigen::MatrixXd &M)
{
    const int n = (int)polygon.rows(); // n == 3
    if (n != 3)
        return;
    M.resize(3, 3);

    Eigen::MatrixXd M_F, P, V;
    M_F.resize(45, 45);
    M_F.setZero();
    P.resize(45, 3);
    V.resize(45, 3);

    //construct the prolongation matrix;
    //P.row(1) << 1., 2., 3.;
    P.row(0) = Eigen::Vector3d(1., 0., 0.);
    P.row(1) = Eigen::Vector3d(0., 1., 0.);
    P.row(2) = Eigen::Vector3d(0., 0., 1.);

    P.row(3) = Eigen::Vector3d(7. / 8., 1. / 8., 0.);
    P.row(4) = Eigen::Vector3d(0.75, 0.25, 0.);
    P.row(5) = Eigen::Vector3d(5. / 8., 3. / 8., 0.);
    P.row(6) = Eigen::Vector3d(0.5, 0.5, 0.); //axis
    P.row(7) = Eigen::Vector3d(3. / 8., 5. / 8., 0.);
    P.row(8) = Eigen::Vector3d(0.25, 0.75, 0.);
    P.row(9) = Eigen::Vector3d(1. / 8., 7. / 8., 0.);

    P.row(10) = Eigen::Vector3d(0., 7. / 8., 1. / 8.);
    P.row(11) = Eigen::Vector3d(0., 0.75, 0.25);
    P.row(12) = Eigen::Vector3d(0., 5. / 8., 3. / 8.);
    P.row(13) = Eigen::Vector3d(0., 0.5, 0.5); //axis
    P.row(14) = Eigen::Vector3d(0., 3. / 8., 5. / 8.);
    P.row(15) = Eigen::Vector3d(0., 0.25, 0.75);
    P.row(16) = Eigen::Vector3d(0., 1. / 8., 7. / 8.);

    P.row(17) = Eigen::Vector3d(1. / 8., 0., 7. / 8.);
    P.row(18) = Eigen::Vector3d(0.25, 0., 0.75);
    P.row(19) = Eigen::Vector3d(3. / 8., 0., 5. / 8.);
    P.row(20) = Eigen::Vector3d(0.5, 0., 0.5); //axis
    P.row(21) = Eigen::Vector3d(5. / 8., 0., 3. / 8.);
    P.row(22) = Eigen::Vector3d(0.75, 0., 0.25);
    P.row(23) = Eigen::Vector3d(7. / 8., 0., 1. / 8.);

    P.row(24) = Eigen::Vector3d(0.75, 1. / 8., 1. / 8.);
    P.row(25) = Eigen::Vector3d(5. / 8., 0.25, 1. / 8.);
    P.row(26) = Eigen::Vector3d(0.5, 3. / 8., 1. / 8.);
    P.row(27) = Eigen::Vector3d(3. / 8., 0.5, 1. / 8.);
    P.row(28) = Eigen::Vector3d(0.25, 5. / 8., 1. / 8.);
    P.row(29) = Eigen::Vector3d(1. / 8., 0.75, 1. / 8.);
    P.row(30) = Eigen::Vector3d(1. / 8., 5. / 8., 0.25);

    P.row(31) = Eigen::Vector3d(1. / 8., 0.5, 3. / 8.);
    P.row(32) = Eigen::Vector3d(1. / 8., 3. / 8., 0.5);
    P.row(33) = Eigen::Vector3d(1. / 8., 0.25, 5. / 8.);
    P.row(34) = Eigen::Vector3d(1. / 8., 1. / 8., 0.75);
    P.row(35) = Eigen::Vector3d(0.25, 1. / 8., 5. / 8.);
    P.row(36) = Eigen::Vector3d(3. / 8., 1. / 8., 0.5);
    P.row(37) = Eigen::Vector3d(0.5, 1. / 8., 3. / 8.);

    P.row(38) = Eigen::Vector3d(5. / 8., 1. / 8., 0.25);
    P.row(39) = Eigen::Vector3d(0.5, 0.25, 0.25);
    P.row(40) = Eigen::Vector3d(3. / 8., 3. / 8., 0.25);
    P.row(41) = Eigen::Vector3d(0.25, 0.5, 0.25);
    P.row(42) = Eigen::Vector3d(0.25, 3. / 8., 3. / 8.);
    P.row(43) = Eigen::Vector3d(0.25, 0.25, 0.5);
    P.row(44) = Eigen::Vector3d(3. / 8., 0.25, 3. / 8.);

    //construct the vertex matrix;
    //V.row(0) = polygon.row(0);
    //V.row(1) = polygon.row(1);
    //V.row(2) = polygon.row(2);

    // compute position of virtual vertex
    // phong tessellation
    Eigen::Vector3d vvertex;
    for (int i = 3; i < 45; i++)
    {
        vvertex = polygon.transpose() * Eigen::Vector3d(P.row(i));
        get_vvertex_hat(polygon, normals, Eigen::Vector3d(P.row(i)), vvertex);
        V.row(i) = vvertex;
    }

    //construct the vertex matrix;
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    //construct refine stiffness matrix;
    // 108 edges
    M_F(0, 3) = pmp::triangle_area(V.row(0), V.row(3), V.row(23)) / 12;
    M_F(0, 23) = pmp::triangle_area(V.row(0), V.row(3), V.row(23)) / 12;

    M_F(1, 9) = pmp::triangle_area(V.row(1), V.row(9), V.row(10)) / 12;
    M_F(1, 10) = pmp::triangle_area(V.row(1), V.row(9), V.row(10)) / 12;

    M_F(2, 16) = pmp::triangle_area(V.row(2), V.row(16), V.row(17)) / 12;
    M_F(2, 17) = pmp::triangle_area(V.row(2), V.row(16), V.row(17)) / 12;

    M_F(3, 4) = pmp::triangle_area(V.row(3), V.row(4), V.row(24)) / 12;
    M_F(3, 23) = (pmp::triangle_area(V.row(0), V.row(3), V.row(23)) +
                  pmp::triangle_area(V.row(3), V.row(23), V.row(24))) /
                 12;
    M_F(3, 24) = (pmp::triangle_area(V.row(3), V.row(4), V.row(24)) +
                  pmp::triangle_area(V.row(3), V.row(23), V.row(24))) /
                 12;

    M_F(4, 5) = pmp::triangle_area(V.row(4), V.row(5), V.row(25)) / 12;
    M_F(4, 24) = (pmp::triangle_area(V.row(3), V.row(4), V.row(24)) +
                  pmp::triangle_area(V.row(4), V.row(24), V.row(25))) /
                 12;
    M_F(4, 25) = (pmp::triangle_area(V.row(4), V.row(5), V.row(25)) +
                  pmp::triangle_area(V.row(4), V.row(24), V.row(25))) /
                 12;

    M_F(5, 6) = pmp::triangle_area(V.row(5), V.row(6), V.row(26)) / 12;
    M_F(5, 25) = (pmp::triangle_area(V.row(4), V.row(5), V.row(25)) +
                  pmp::triangle_area(V.row(5), V.row(25), V.row(26))) /
                 12;
    M_F(5, 26) = (pmp::triangle_area(V.row(5), V.row(6), V.row(26)) +
                  pmp::triangle_area(V.row(5), V.row(25), V.row(26))) /
                 12;

    M_F(6, 7) = pmp::triangle_area(V.row(6), V.row(7), V.row(27)) / 12;
    M_F(6, 26) = (pmp::triangle_area(V.row(5), V.row(6), V.row(26)) +
                  pmp::triangle_area(V.row(6), V.row(26), V.row(27))) /
                 12;
    M_F(6, 27) = (pmp::triangle_area(V.row(6), V.row(7), V.row(27)) +
                  pmp::triangle_area(V.row(6), V.row(26), V.row(27))) /
                 12;

    M_F(7, 8) = pmp::triangle_area(V.row(7), V.row(8), V.row(28)) / 12;
    M_F(7, 27) = (pmp::triangle_area(V.row(6), V.row(7), V.row(27)) +
                  pmp::triangle_area(V.row(7), V.row(27), V.row(28))) /
                 12;
    M_F(7, 28) = (pmp::triangle_area(V.row(7), V.row(8), V.row(28)) +
                  pmp::triangle_area(V.row(7), V.row(27), V.row(28))) /
                 12;

    M_F(8, 9) = pmp::triangle_area(V.row(8), V.row(9), V.row(29)) / 12;
    M_F(8, 28) = (pmp::triangle_area(V.row(7), V.row(8), V.row(28)) +
                  pmp::triangle_area(V.row(8), V.row(28), V.row(29))) /
                 12;
    M_F(8, 29) = (pmp::triangle_area(V.row(8), V.row(9), V.row(29)) +
                  pmp::triangle_area(V.row(8), V.row(28), V.row(29))) /
                 12;

    M_F(9, 10) = (pmp::triangle_area(V.row(1), V.row(9), V.row(10)) +
                  pmp::triangle_area(V.row(9), V.row(10), V.row(29))) /
                 12;
    M_F(9, 29) = (pmp::triangle_area(V.row(8), V.row(9), V.row(29)) +
                  pmp::triangle_area(V.row(9), V.row(10), V.row(29))) /
                 12;

    M_F(10, 11) = pmp::triangle_area(V.row(10), V.row(11), V.row(29)) / 12;
    M_F(10, 29) = (pmp::triangle_area(V.row(10), V.row(11), V.row(29)) +
                   pmp::triangle_area(V.row(9), V.row(10), V.row(29))) /
                  12;

    M_F(11, 12) = pmp::triangle_area(V.row(11), V.row(12), V.row(30)) / 12;
    M_F(11, 29) = (pmp::triangle_area(V.row(10), V.row(11), V.row(29)) +
                   pmp::triangle_area(V.row(11), V.row(29), V.row(30))) /
                  12;
    M_F(11, 30) = (pmp::triangle_area(V.row(11), V.row(12), V.row(30)) +
                   pmp::triangle_area(V.row(11), V.row(29), V.row(30))) /
                  12;

    M_F(12, 13) = pmp::triangle_area(V.row(12), V.row(13), V.row(31)) / 12;
    M_F(12, 30) = (pmp::triangle_area(V.row(11), V.row(12), V.row(30)) +
                   pmp::triangle_area(V.row(12), V.row(30), V.row(31))) /
                  12;
    M_F(12, 31) = (pmp::triangle_area(V.row(12), V.row(13), V.row(31)) +
                   pmp::triangle_area(V.row(12), V.row(30), V.row(31))) /
                  12;

    M_F(13, 14) = pmp::triangle_area(V.row(13), V.row(14), V.row(31)) / 12;
    M_F(13, 31) = (pmp::triangle_area(V.row(12), V.row(13), V.row(31)) +
                   pmp::triangle_area(V.row(13), V.row(31), V.row(32))) /
                  12;
    M_F(13, 32) = (pmp::triangle_area(V.row(13), V.row(14), V.row(32)) +
                   pmp::triangle_area(V.row(13), V.row(31), V.row(32))) /
                  12;

    M_F(14, 15) = pmp::triangle_area(V.row(14), V.row(15), V.row(33)) / 12;
    M_F(14, 32) = (pmp::triangle_area(V.row(13), V.row(14), V.row(32)) +
                   pmp::triangle_area(V.row(14), V.row(32), V.row(33))) /
                  12;
    M_F(14, 33) = (pmp::triangle_area(V.row(14), V.row(15), V.row(33)) +
                   pmp::triangle_area(V.row(14), V.row(32), V.row(33))) /
                  12;

    M_F(15, 16) = pmp::triangle_area(V.row(15), V.row(16), V.row(34)) / 12;
    M_F(15, 33) = (pmp::triangle_area(V.row(14), V.row(15), V.row(33)) +
                   pmp::triangle_area(V.row(15), V.row(33), V.row(34))) /
                  12;
    M_F(15, 34) = (pmp::triangle_area(V.row(15), V.row(16), V.row(34)) +
                   pmp::triangle_area(V.row(15), V.row(33), V.row(34))) /
                  12;

    M_F(16, 17) = (pmp::triangle_area(V.row(2), V.row(16), V.row(17)) +
                   pmp::triangle_area(V.row(16), V.row(17), V.row(34))) /
                  12;
    M_F(16, 34) = (pmp::triangle_area(V.row(15), V.row(16), V.row(34)) +
                   pmp::triangle_area(V.row(16), V.row(17), V.row(34))) /
                  12;

    M_F(17, 18) = pmp::triangle_area(V.row(17), V.row(18), V.row(34)) / 12;
    M_F(17, 34) = (pmp::triangle_area(V.row(17), V.row(18), V.row(34)) +
                   pmp::triangle_area(V.row(16), V.row(17), V.row(34))) /
                  12;

    M_F(18, 19) = pmp::triangle_area(V.row(18), V.row(19), V.row(35)) / 12;
    M_F(18, 34) = (pmp::triangle_area(V.row(17), V.row(18), V.row(34)) +
                   pmp::triangle_area(V.row(18), V.row(34), V.row(35))) /
                  12;
    M_F(18, 35) = (pmp::triangle_area(V.row(18), V.row(19), V.row(35)) +
                   pmp::triangle_area(V.row(18), V.row(34), V.row(35))) /
                  12;

    M_F(19, 20) = pmp::triangle_area(V.row(19), V.row(20), V.row(36)) / 12;
    M_F(19, 35) = (pmp::triangle_area(V.row(18), V.row(19), V.row(35)) +
                   pmp::triangle_area(V.row(19), V.row(35), V.row(36))) /
                  12;
    M_F(19, 36) = (pmp::triangle_area(V.row(19), V.row(20), V.row(36)) +
                   pmp::triangle_area(V.row(19), V.row(35), V.row(36))) /
                  12;

    M_F(20, 21) = pmp::triangle_area(V.row(20), V.row(21), V.row(37)) / 12;
    M_F(20, 36) = (pmp::triangle_area(V.row(19), V.row(20), V.row(36)) +
                   pmp::triangle_area(V.row(20), V.row(36), V.row(37))) /
                  12;
    M_F(20, 37) = (pmp::triangle_area(V.row(20), V.row(21), V.row(37)) +
                   pmp::triangle_area(V.row(20), V.row(36), V.row(37))) /
                  12;

    M_F(21, 22) = pmp::triangle_area(V.row(21), V.row(22), V.row(38)) / 12;
    M_F(21, 37) = (pmp::triangle_area(V.row(20), V.row(21), V.row(37)) +
                   pmp::triangle_area(V.row(21), V.row(37), V.row(38))) /
                  12;
    M_F(21, 38) = (pmp::triangle_area(V.row(21), V.row(22), V.row(38)) +
                   pmp::triangle_area(V.row(21), V.row(37), V.row(38))) /
                  12;

    M_F(22, 23) = pmp::triangle_area(V.row(22), V.row(23), V.row(24)) / 12;
    M_F(22, 24) = (pmp::triangle_area(V.row(22), V.row(23), V.row(24)) +
                   pmp::triangle_area(V.row(22), V.row(24), V.row(38))) /
                  12;
    M_F(22, 38) = (pmp::triangle_area(V.row(21), V.row(22), V.row(38)) +
                   pmp::triangle_area(V.row(22), V.row(24), V.row(38))) /
                  12;

    M_F(23, 24) = (pmp::triangle_area(V.row(22), V.row(23), V.row(24)) +
                   pmp::triangle_area(V.row(3), V.row(23), V.row(24))) /
                  12;

    M_F(24, 25) = (pmp::triangle_area(V.row(4), V.row(24), V.row(25)) +
                   pmp::triangle_area(V.row(24), V.row(25), V.row(38))) /
                  12;
    M_F(24, 38) = (pmp::triangle_area(V.row(22), V.row(24), V.row(38)) +
                   pmp::triangle_area(V.row(24), V.row(25), V.row(38))) /
                  12;

    M_F(25, 26) = (pmp::triangle_area(V.row(5), V.row(25), V.row(26)) +
                   pmp::triangle_area(V.row(25), V.row(26), V.row(39))) /
                  12;
    M_F(25, 38) = (pmp::triangle_area(V.row(25), V.row(38), V.row(39)) +
                   pmp::triangle_area(V.row(24), V.row(25), V.row(38))) /
                  12;
    M_F(25, 39) = (pmp::triangle_area(V.row(25), V.row(38), V.row(39)) +
                   pmp::triangle_area(V.row(25), V.row(26), V.row(39))) /
                  12;

    M_F(26, 27) = (pmp::triangle_area(V.row(6), V.row(26), V.row(27)) +
                   pmp::triangle_area(V.row(26), V.row(27), V.row(40))) /
                  12;
    M_F(26, 39) = (pmp::triangle_area(V.row(26), V.row(39), V.row(40)) +
                   pmp::triangle_area(V.row(25), V.row(26), V.row(39))) /
                  12;
    M_F(26, 40) = (pmp::triangle_area(V.row(26), V.row(39), V.row(40)) +
                   pmp::triangle_area(V.row(26), V.row(27), V.row(40))) /
                  12;

    M_F(27, 28) = (pmp::triangle_area(V.row(7), V.row(27), V.row(28)) +
                   pmp::triangle_area(V.row(27), V.row(28), V.row(41))) /
                  12;
    M_F(27, 40) = (pmp::triangle_area(V.row(27), V.row(40), V.row(41)) +
                   pmp::triangle_area(V.row(26), V.row(27), V.row(40))) /
                  12;
    M_F(27, 41) = (pmp::triangle_area(V.row(27), V.row(40), V.row(41)) +
                   pmp::triangle_area(V.row(27), V.row(28), V.row(41))) /
                  12;

    M_F(28, 29) = (pmp::triangle_area(V.row(8), V.row(28), V.row(29)) +
                   pmp::triangle_area(V.row(28), V.row(29), V.row(30))) /
                  12;
    M_F(28, 30) = (pmp::triangle_area(V.row(28), V.row(30), V.row(41)) +
                   pmp::triangle_area(V.row(28), V.row(29), V.row(30))) /
                  12;
    M_F(28, 41) = (pmp::triangle_area(V.row(28), V.row(30), V.row(41)) +
                   pmp::triangle_area(V.row(27), V.row(28), V.row(41))) /
                  12;

    M_F(29, 30) = (pmp::triangle_area(V.row(11), V.row(29), V.row(30)) +
                   pmp::triangle_area(V.row(28), V.row(29), V.row(30))) /
                  12;

    M_F(30, 31) = (pmp::triangle_area(V.row(12), V.row(30), V.row(31)) +
                   pmp::triangle_area(V.row(30), V.row(31), V.row(41))) /
                  12;
    M_F(30, 41) = (pmp::triangle_area(V.row(28), V.row(30), V.row(41)) +
                   pmp::triangle_area(V.row(30), V.row(31), V.row(41))) /
                  12;

    M_F(31, 32) = (pmp::triangle_area(V.row(13), V.row(31), V.row(32)) +
                   pmp::triangle_area(V.row(31), V.row(32), V.row(42))) /
                  12;
    M_F(31, 41) = (pmp::triangle_area(V.row(31), V.row(41), V.row(42)) +
                   pmp::triangle_area(V.row(30), V.row(31), V.row(41))) /
                  12;
    M_F(31, 42) = (pmp::triangle_area(V.row(31), V.row(41), V.row(42)) +
                   pmp::triangle_area(V.row(31), V.row(32), V.row(42))) /
                  12;

    M_F(32, 33) = (pmp::triangle_area(V.row(14), V.row(32), V.row(33)) +
                   pmp::triangle_area(V.row(32), V.row(33), V.row(43))) /
                  12;
    M_F(32, 42) = (pmp::triangle_area(V.row(32), V.row(42), V.row(43)) +
                   pmp::triangle_area(V.row(31), V.row(32), V.row(42))) /
                  12;
    M_F(32, 43) = (pmp::triangle_area(V.row(32), V.row(42), V.row(43)) +
                   pmp::triangle_area(V.row(32), V.row(33), V.row(43))) /
                  12;

    M_F(33, 34) = (pmp::triangle_area(V.row(15), V.row(33), V.row(34)) +
                   pmp::triangle_area(V.row(33), V.row(34), V.row(35))) /
                  12;
    M_F(33, 35) = (pmp::triangle_area(V.row(33), V.row(35), V.row(43)) +
                   pmp::triangle_area(V.row(33), V.row(34), V.row(35))) /
                  12;
    M_F(33, 43) = (pmp::triangle_area(V.row(33), V.row(35), V.row(43)) +
                   pmp::triangle_area(V.row(32), V.row(33), V.row(43))) /
                  12;

    M_F(34, 35) = (pmp::triangle_area(V.row(18), V.row(34), V.row(35)) +
                   pmp::triangle_area(V.row(33), V.row(34), V.row(35))) /
                  12;

    M_F(35, 36) = (pmp::triangle_area(V.row(19), V.row(35), V.row(36)) +
                   pmp::triangle_area(V.row(35), V.row(36), V.row(43))) /
                  12;
    M_F(35, 43) = (pmp::triangle_area(V.row(33), V.row(35), V.row(43)) +
                   pmp::triangle_area(V.row(35), V.row(36), V.row(43))) /
                  12;

    M_F(36, 37) = (pmp::triangle_area(V.row(20), V.row(36), V.row(37)) +
                   pmp::triangle_area(V.row(36), V.row(37), V.row(44))) /
                  12;
    M_F(36, 43) = (pmp::triangle_area(V.row(36), V.row(43), V.row(44)) +
                   pmp::triangle_area(V.row(35), V.row(36), V.row(43))) /
                  12;
    M_F(36, 44) = (pmp::triangle_area(V.row(36), V.row(43), V.row(44)) +
                   pmp::triangle_area(V.row(36), V.row(37), V.row(44))) /
                  12;

    M_F(37, 38) = (pmp::triangle_area(V.row(21), V.row(37), V.row(38)) +
                   pmp::triangle_area(V.row(37), V.row(38), V.row(39))) /
                  12;
    M_F(37, 39) = (pmp::triangle_area(V.row(37), V.row(39), V.row(44)) +
                   pmp::triangle_area(V.row(37), V.row(38), V.row(39))) /
                  12;
    M_F(37, 44) = (pmp::triangle_area(V.row(37), V.row(39), V.row(44)) +
                   pmp::triangle_area(V.row(36), V.row(37), V.row(44))) /
                  12;

    M_F(38, 39) = (pmp::triangle_area(V.row(25), V.row(38), V.row(39)) +
                   pmp::triangle_area(V.row(37), V.row(38), V.row(39))) /
                  12;

    M_F(39, 40) = (pmp::triangle_area(V.row(26), V.row(39), V.row(40)) +
                   pmp::triangle_area(V.row(39), V.row(40), V.row(44))) /
                  12;
    M_F(39, 44) = (pmp::triangle_area(V.row(37), V.row(39), V.row(44)) +
                   pmp::triangle_area(V.row(39), V.row(40), V.row(44))) /
                  12;

    M_F(40, 41) = (pmp::triangle_area(V.row(27), V.row(40), V.row(41)) +
                   pmp::triangle_area(V.row(40), V.row(41), V.row(42))) /
                  12;
    M_F(40, 42) = (pmp::triangle_area(V.row(40), V.row(42), V.row(44)) +
                   pmp::triangle_area(V.row(40), V.row(41), V.row(42))) /
                  12;
    M_F(40, 44) = (pmp::triangle_area(V.row(40), V.row(42), V.row(44)) +
                   pmp::triangle_area(V.row(39), V.row(40), V.row(44))) /
                  12;

    M_F(41, 42) = (pmp::triangle_area(V.row(31), V.row(41), V.row(42)) +
                   pmp::triangle_area(V.row(40), V.row(41), V.row(42))) /
                  12;

    M_F(42, 43) = (pmp::triangle_area(V.row(32), V.row(42), V.row(43)) +
                   pmp::triangle_area(V.row(42), V.row(43), V.row(44))) /
                  12;
    M_F(42, 44) = (pmp::triangle_area(V.row(40), V.row(42), V.row(44)) +
                   pmp::triangle_area(V.row(42), V.row(43), V.row(44))) /
                  12;

    M_F(43, 44) = (pmp::triangle_area(V.row(36), V.row(43), V.row(44)) +
                   pmp::triangle_area(V.row(42), V.row(43), V.row(44))) /
                  12;

    for (int j = 0; j < 45; j++) //transpose
    {
        for (int i = j + 1; i < 45; i++)
        {
            M_F(i, j) = M_F(j, i);
        }
    }

    for (int i = 0; i < 45; i++)
    {
        double w = 0;
        for (int j = 0; j < 45; j++)
        {
            w += M_F(i, j);
        }
        M_F(i, i) = w;
    }

    // sandwiching with (local) restriction and prolongation matrix
    M = P.transpose() * M_F * P;
    //std::cout << M_F << std::endl;
}

//----------------------------------------------------------------------------------

void setup_quadratic_mass_matrix(const Eigen::MatrixXd &polygon,
                                 const std::vector<Normal> &normals,
                                 Eigen::MatrixXd &M)
{
    const int n = (int)polygon.rows(); // n == 3
    if (n != 3)
        return;
    M.resize(3, 3);

    Eigen::MatrixXd M_F, P, V, P0;
    M_F.resize(9, 9);
    M_F.setZero();
    P.resize(9, 6);
    P0.resize(6, 3);
    V.resize(9, 3);

    //construct the prolongation matrix;
    P << 1., 0., 0., 0., 0., 0., 
        0., 1., 0., 0., 0., 0.,
        0., 0., 1., 0., 0., 0.,
        0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 1., 0.,
        0., 0., 0., 0., 0., 1.,
        //0.5, 0.25, 0.25, 0., 0., 0., 
        //0.25, 0.5, 0.25, 0., 0., 0., 
        //0.25, 0.25, 0.5, 0., 0., 0.;
        0., 0., 0., 0.5, 0.5, 0.,
        0., 0., 0., 0., 0.5, 0.5, 
        0., 0., 0., 0.5, 0., 0.5;

    //construct the vertex matrix，2*n vertex
    V.row(0) = polygon.row(0);
    V.row(1) = polygon.row(1);
    V.row(2) = polygon.row(2);

    V.row(3) = 0.5 * (V.row(0) + V.row(1));
    V.row(4) = 0.5 * (V.row(1) + V.row(2));
    V.row(5) = 0.5 * (V.row(0) + V.row(2));
    //n virtual vertex;
    V.row(6) = 0.5 * (V.row(3) + V.row(5));
    V.row(7) = 0.5 * (V.row(3) + V.row(4));
    V.row(8) = 0.5 * (V.row(4) + V.row(5));

    //construct refine mass matrix;
    M_F(0, 3) = pmp::triangle_area(V.row(0), V.row(3), V.row(6)) / 12;
    M_F(0, 5) = pmp::triangle_area(V.row(0), V.row(5), V.row(6)) / 12;
    M_F(0, 6) = (pmp::triangle_area(V.row(0), V.row(3), V.row(6)) +
                 pmp::triangle_area(V.row(0), V.row(5), V.row(6))) /
                12;

    M_F(1, 3) = pmp::triangle_area(V.row(1), V.row(3), V.row(7)) / 12;
    M_F(1, 4) = pmp::triangle_area(V.row(1), V.row(4), V.row(7)) / 12;
    M_F(1, 7) = (pmp::triangle_area(V.row(1), V.row(3), V.row(7)) +
                 pmp::triangle_area(V.row(1), V.row(4), V.row(7))) /
                12;

    M_F(2, 4) = pmp::triangle_area(V.row(2), V.row(4), V.row(8)) / 12;
    M_F(2, 5) = pmp::triangle_area(V.row(2), V.row(5), V.row(8)) / 12;
    M_F(2, 8) = (pmp::triangle_area(V.row(2), V.row(4), V.row(8)) +
                 pmp::triangle_area(V.row(2), V.row(5), V.row(8))) /
                12;

    M_F(3, 6) = (pmp::triangle_area(V.row(0), V.row(3), V.row(6)) +
                 pmp::triangle_area(V.row(3), V.row(6), V.row(7))) /
                12;
    M_F(3, 7) = (pmp::triangle_area(V.row(1), V.row(3), V.row(7)) +
                 pmp::triangle_area(V.row(3), V.row(6), V.row(7))) /
                12;

    M_F(4, 7) = (pmp::triangle_area(V.row(1), V.row(4), V.row(7)) +
                 pmp::triangle_area(V.row(4), V.row(7), V.row(8))) /
                12;
    M_F(4, 8) = (pmp::triangle_area(V.row(4), V.row(7), V.row(8)) +
                 pmp::triangle_area(V.row(2), V.row(4), V.row(8))) /
                12;

    M_F(5, 6) = (pmp::triangle_area(V.row(0), V.row(5), V.row(6)) +
                 pmp::triangle_area(V.row(5), V.row(6), V.row(8))) /
                12;
    M_F(5, 8) = (pmp::triangle_area(V.row(2), V.row(5), V.row(8)) +
                 pmp::triangle_area(V.row(5), V.row(6), V.row(8))) /
                12;

    M_F(6, 7) = (pmp::triangle_area(V.row(6), V.row(7), V.row(8)) +
                 pmp::triangle_area(V.row(3), V.row(6), V.row(7))) /
                12;
    M_F(6, 8) = (pmp::triangle_area(V.row(6), V.row(7), V.row(8)) +
                 pmp::triangle_area(V.row(5), V.row(6), V.row(8))) /
                12;

    M_F(7, 8) = (pmp::triangle_area(V.row(6), V.row(7), V.row(8)) +
                 pmp::triangle_area(V.row(4), V.row(7), V.row(8))) /
                12;

    for (int j = 0; j < 9; j++) //transpose
    {
        for (int i = j + 1; i < 9; i++)
        {
            M_F(i, j) = M_F(j, i);
        }
    }

    for (int i = 0; i < 9; i++)
    {
        double w = 0;
        for (int j = 0; j < 9; j++)
        {
            w += M_F(i, j);
        }
        M_F(i, i) = w;
    }

    P0 << 1., 0., 0., 0., 1., 0., 
        0., 0., 1., 0.5, 0.5, 0., 
        0., 0.5, 0.5, 0.5, 0., 0.5;

    // sandwiching with (local) restriction and prolongation matrix
    M = (P0.transpose() * (P.transpose() * M_F * P) * P0);
    //std::cout << S_F << std::endl;
}

//----------------------------------------------------------------------------------
    void lump_matrix(SparseMatrix &D)
{
    std::vector<Triplet> triplets;
    triplets.reserve(D.rows() * 6);

    for (int k = 0; k < D.outerSize(); ++k)
    {
        for (SparseMatrix::InnerIterator it(D, k); it; ++it)
        {
            triplets.emplace_back(it.row(), it.row(), it.value());
        }
    }

    D.setFromTriplets(triplets.begin(), triplets.end());
}

//----------------------------------------------------------------------------------

void setup_prolongation_matrix(const SurfaceMesh &mesh, SparseMatrix &A)
{
    auto area_weights = mesh.get_face_property<Eigen::VectorXd>("f:weights");

    const unsigned int nv = mesh.n_vertices();
    const unsigned int nf = mesh.n_faces();

    std::vector<Triplet> tripletsA;

    for (auto v : mesh.vertices())
    {
        tripletsA.emplace_back(v.idx(), v.idx(), 1.0);
    }

    unsigned int j = 0;
    for (auto f : mesh.faces())
    {
        const Eigen::VectorXd& w = area_weights[f];

        unsigned int i = 0;
        for (auto v : mesh.vertices(f))
        {
            tripletsA.emplace_back(nv + j, v.idx(), w(i));
            i++;
        }
        j++;
    }

    // build sparse matrix from triplets
    A.resize(nv + nf, nv);
    A.setFromTriplets(tripletsA.begin(), tripletsA.end());
}

//-----------------------------------------------------------------------------

Scalar mesh_area(const SurfaceMesh &mesh)
{
    Scalar area(0);
    for (auto f : mesh.faces())
    {
        area += face_area(mesh, f);
    }
    return area;
}

//-----------------------------------------------------------------------------

double face_area(const SurfaceMesh &mesh, Face f)
{
#if 0
    // vector area of a polygon
    Normal n(0, 0, 0);
    for (auto h : mesh.halfedges(f))
        n += cross(mesh.position(mesh.from_vertex(h)), 
                   mesh.position(mesh.to_vertex(h)));
    return 0.5 * norm(n);
#else
    double a = 0.0;
    Point C = centroid(mesh, f);
    Point Q, R;
    for (auto h : mesh.halfedges(f)) {
        Q = mesh.position(mesh.from_vertex(h));
        R = mesh.position(mesh.to_vertex(h));
        a += pmp::triangle_area(C, Q, R);
    }
    return a;
#endif
}

//-----------------------------------------------------------------------------

Point area_weighted_centroid(const SurfaceMesh &mesh)
{
    Point center(0, 0, 0), c;
    Scalar area(0), a;
    for (auto f : mesh.faces())
    {
        int count = 0;
        c = Point(0, 0, 0);
        for (auto v : mesh.vertices(f))
        {
            c += mesh.position(v);
            count++;
        }
        c /= (Scalar)count;
        a = (Scalar)face_area(mesh, f);
        area += a;
        center += a * c;
    }
    return center /= area;
}

//-----------------------------------------------------------------------------

Eigen::Vector3d gradient_hat_function(const Point& i, const Point& j, const Point& k)
{
    Point base, site, grad;
    Eigen::Vector3d gradient;
    double area;
    area = triangle_area(i, j, k);
    site = i - j;
    base = k - j;
    grad = site - (dot(site, base) / norm(base)) * base / norm(base);//不用叉积
    if (area < eps)
    {
        gradient = Eigen::Vector3d(0, 0, 0);
    }
    else
    {
        grad = norm(base) * grad / norm(grad);
        gradient = Eigen::Vector3d(grad[0], grad[1], grad[2]) / (2.0 * area);
    }

    return gradient;
}

//-----------------------------------------------------------------------------

void setup_gradient_matrix(const SurfaceMesh &mesh, SparseMatrix &G)
{
    //SparseMatrix A;
    //setup_prolongation_matrix(mesh, A);

    const unsigned int nv = mesh.n_vertices();
    //const unsigned int nf = mesh.n_faces();
    Point p, p0, p1;
    Vertex v,v0, v1;
    int nr_triangles = 0;
    int k = 0;
    //auto area_points = mesh.get_face_property<Point>("f:point");
    Eigen::Vector3d gradient_p, gradient_p0, gradient_p1;
    // nonzero elements of G as triplets: (row, column, value)
    std::vector<Triplet> triplets;

    for (Face f : mesh.faces())
    {
        nr_triangles += mesh.valence(f);
        //p = area_points[f];
        for (auto h : mesh.halfedges(f))
        {
            v = mesh.to_vertex(mesh.next_halfedge(h));
            v0 = mesh.from_vertex(h);
            v1 = mesh.to_vertex(h);

            p = mesh.position(v);
            p0 = mesh.position(v0);
            p1 = mesh.position(v1);

            gradient_p = gradient_hat_function(p, p0, p1);
            gradient_p0 = gradient_hat_function(p0, p1, p);
            gradient_p1 = gradient_hat_function(p1, p, p0);

            for (int j = 0; j < 3; j++)
            {
                triplets.emplace_back(3 * k + j, v.idx(), gradient_p(j));
                triplets.emplace_back(3 * k + j, v0.idx(), gradient_p0(j));
                triplets.emplace_back(3 * k + j, v1.idx(), gradient_p1(j));
            }
            k++;
        }
    }

    G.resize(3 * nr_triangles, nv);
    G.setFromTriplets(triplets.begin(), triplets.end());
    //G = G * A;
}

//-----------------------------------------------------------------------------

void setup_divergence_matrix(const SurfaceMesh &mesh, SparseMatrix &Gt)
{
    SparseMatrix G, M;
    setup_gradient_matrix(mesh, G);
    setup_gradient_mass_matrix(mesh, M);
    Gt = -G.transpose() * M;
}

//-----------------------------------------------------------------------------

void setup_gradient_mass_matrix(const SurfaceMesh &mesh,
                                Eigen::SparseMatrix<double> &M)
{
    //auto area_points = mesh.get_face_property<Point>("f:point");
    double area;
    std::vector<Eigen::Triplet<double>> triplets;
    int valence, idx, c = 0;
    for (auto f : mesh.faces())
    {
        valence = mesh.valence(f);
        int i = 0;
        for (auto h : mesh.halfedges(f))
        {
            Point p = mesh.position(mesh.to_vertex(mesh.next_halfedge(h)));
            Point p0 = mesh.position(mesh.from_vertex(h));
            Point p1 = mesh.position(mesh.to_vertex(h));
            area = triangle_area(p0, p1, p);
            for (int j = 0; j < 3; j++)
            {
                idx = c + 3 * i + j;
                triplets.emplace_back(idx, idx, area);
            }
            i++;
        }
        c += valence * 3;
    }
    M.resize(c, c);

    M.setFromTriplets(triplets.begin(), triplets.end());
}

//-----------------------------------------------------------------------------

void setup_virtual_vertices(SurfaceMesh &mesh)
{
    auto area_points = mesh.get_face_property<Point>("f:point");
    auto area_weights = mesh.get_face_property<Eigen::VectorXd>("f:weights");

    Eigen::VectorXd w;
    Eigen::MatrixXd poly;

    for (Face f : mesh.faces())
    {
        const int n = mesh.valence(f);
        poly.resize(n, 3);
        int i = 0;

        std::vector<Normal> normals; // polygon normals
        for (Vertex v : mesh.vertices(f))
        {
            poly.row(i) = (Eigen::Vector3d) mesh.position(v);
            pre_compute_normals(mesh);
            normals.push_back(pre_normals[v.idx()]);
            i++;
        }


        compute_virtual_vertex(poly, w);
        Eigen::Vector3d vvertex =  poly.transpose() * w;
        get_vvertex_hat(poly, normals, w, vvertex);
        area_points[f] = vvertex;
        //area_points[f] = poly.transpose() * w;
        area_weights[f] = w;
    }
}

//-----------------------------------------------------------------------------
/*
void compute_virtual_vertex(const Eigen::MatrixXd &poly, Eigen::VectorXd &weights)
{
    int val = poly.rows();
    Eigen::MatrixXd J(val, val);
    Eigen::VectorXd b(val);
    weights.resize(val);

    for (int i = 0; i < val; i++)
    {
        Eigen::Vector3d pk = poly.row(i);

        double Bk1_d2 = 0.0;
        double Bk1_d1 = 0.0;

        double Bk2_d0 = 0.0;
        double Bk2_d2 = 0.0;

        double Bk3_d0 = 0.0;
        double Bk3_d1 = 0.0;

        double CBk = 0.0;
        Eigen::Vector3d d = Eigen::MatrixXd::Zero(3, 1);

        for (int j = 0; j < val; j++)
        {
            Eigen::Vector3d pi = poly.row(j);
            Eigen::Vector3d pj = poly.row((j + 1) % val);
            d = pi - pj;

            double Bik1 = d(1) * pk(2) - d(2) * pk(1);
            double Bik2 = d(2) * pk(0) - d(0) * pk(2);
            double Bik3 = d(0) * pk(1) - d(1) * pk(0);

            double Ci1 = d(1) * pi(2) - d(2) * pi(1);
            double Ci2 = d(2) * pi(0) - d(0) * pi(2);
            double Ci3 = d(0) * pi(1) - d(1) * pi(0);

            Bk1_d1 += d(1) * Bik1;
            Bk1_d2 += d(2) * Bik1;

            Bk2_d0 += d(0) * Bik2;
            Bk2_d2 += d(2) * Bik2;

            Bk3_d0 += d(0) * Bik3;
            Bk3_d1 += d(1) * Bik3;

            CBk += Ci1 * Bik1 + Ci2 * Bik2 + Ci3 * Bik3;
        }
        for (int k = 0; k < val; k++)
        {
            Eigen::Vector3d xj = poly.row(k);
            J(i, k) = 0.5 * (xj(2) * Bk1_d1 - xj(1) * Bk1_d2 + xj(0) * Bk2_d2 -
                             xj(2) * Bk2_d0 + xj(1) * Bk3_d0 - xj(0) * Bk3_d1);
        }
        b(i) = 0.5 * CBk;
    }

    Eigen::MatrixXd M(val + 1, val);
    M.block(0, 0, val, val) = 4 * J;
    M.block(val, 0, 1, val).setOnes();

    Eigen::VectorXd b_(val + 1);
    b_.block(0, 0, val, 1) = 4 * b;

    b_(val) = 1.;

    weights = M.completeOrthogonalDecomposition().solve(b_).topRows(val);
}
*/
//-----------------------------------------------------------------------------

//=============================================================================

void barycentric_Coordinate_Interpolation(
    const std::vector<Eigen::Vector3d> &points, Eigen::Vector3d &centroid,
    Eigen::Vector3d &coord)
{
    //先求得三角形质心，然后求得重心插值坐标(两者不相同)
    centroid.setZero();
    for (const Eigen::Vector3d &point : points) {
        centroid += point;
    }
    centroid /= (double)points.size();//找到重心
    
    //重心坐标插值
    Eigen::MatrixXd tmp;//3*2
    tmp.resize(3, 2);
    tmp.col(0) = points[0] - points[2];
    tmp.col(1) = points[1] - points[2];
    //求解一个非齐次线性方程组
    Eigen::Vector2d uv = tmp.fullPivLu().solve(centroid - points[0]);
    coord(0) = 1 - uv(0) - uv(1);//w
    coord(1) = uv(0);//u
    coord(2) = uv(1);//v
    
}
//=============================================================================