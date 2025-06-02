#include <pmp/algorithms/SurfaceNormals.h>

#include "VertexNormal.h"
#include "PolyDiffGeo.h"

using namespace pmp;

Normal VertexNormals::compute_uniform_vertex_normal(const SurfaceMesh& mesh, Vertex v)
{
    Point nn(0, 0, 0);

    if (!mesh.is_isolated(v))
    {
        auto vpoint = mesh.get_vertex_property<Point>("v:point");
        const Point p0 = vpoint[v];

        Normal n;
        Point p1, p2;
        bool is_triangle;

        for (auto h : mesh.halfedges(v))
        {
            if (!mesh.is_boundary(h))
            {
                p1 = vpoint[mesh.to_vertex(h)];
                p1 -= p0;
                p2 = vpoint[mesh.from_vertex(mesh.prev_halfedge(h))];
                p2 -= p0;

                //triangle or polygon normal
                is_triangle = (mesh.next_halfedge(mesh.next_halfedge(mesh.next_halfedge(h))) == h);
                n = is_triangle ? normalize(cross(p1, p2))
                                : SurfaceNormals::compute_face_normal(mesh, mesh.face(h));

                nn += n;//uniform weight
            }
        }

        nn == normalize(nn);
    }
    return nn;
}

Normal VertexNormals::compute_area_vertex_normal(const SurfaceMesh& mesh, Vertex v)
{
    Point nn(0, 0, 0);

    if (!mesh.is_isolated(v))
    {
        auto vpoint = mesh.get_vertex_property<Point>("v:point");
        const Point p0 = vpoint[v];

        Normal n;
        Point p1, p2;
        Scalar cosine, angle, denom;
        bool is_triangle;

        for (auto h : mesh.halfedges(v))
        {
            if (!mesh.is_boundary(h))
            {
                p1 = vpoint[mesh.to_vertex(h)];
                p1 -= p0;
                p2 = vpoint[mesh.from_vertex(mesh.prev_halfedge(h))];
                p2 -= p0;

                // check whether we can robustly compute angle
                denom = sqrt(dot(p1, p1) * dot(p2, p2));
                // exclude 0
                if (denom > std::numeric_limits<Scalar>::min())
                {
                    cosine = dot(p1, p2) / denom;
                    if (cosine < -1.0)
                        cosine = -1.0;
                    else if (cosine > 1.0)
                        cosine = 1.0;
                    angle = std::acos(cosine);//require [-1, 1].return [0, M_PI]

                    // compute triangle or polygon normal
                    is_triangle = (mesh.next_halfedge(mesh.next_halfedge(
                                       mesh.next_halfedge(h))) == h);
                    n = is_triangle ? normalize(cross(p1, p2))
                                    : SurfaceNormals::compute_face_normal(mesh, mesh.face(h));

                    // area weight 
                    n *= face_area(mesh, mesh.face(h));
                    nn += n;
                }
            }
        }

        nn = normalize(nn);
    }
    return nn;
}