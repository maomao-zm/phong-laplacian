//=============================================================================
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================

#include "Viewer.h"
#include "Parameterization.h"
#include "Smoothing.h"
#include "PolyDiffGeo.h"
#include "MeanCurvature.h"
#include "GeodesicsInHeat.h"
#include "EnergySmoothing.h"
#include "BoundarySmoothing.h"
#include "Deformation.h"

#include <pmp/algorithms/DifferentialGeometry.h>
#include <pmp/algorithms/SurfaceSubdivision.h>

#include <imgui.h>
#include <random>

//=============================================================================

using namespace pmp;

struct State
{
    Eigen::Vector3d hPosition;
    unsigned int hIdx = 0;
    unsigned int nRoi = 0;
    std::vector<int> roiIndices;
    //VertexProperty<Color> colors;
};

State s;

//=============================================================================

void Viewer::load_mesh(const char *filename)
{
    MeshViewer::load_mesh(filename);
    set_draw_mode("Hidden Line");
}

//----------------------------------------------------------------------------

void Viewer::process_imgui()
{
    // add standard mesh info stuff
    pmp::MeshViewer::process_imgui();

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Laplacian Setting", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int flag = 2;

        ImGui::PushItemWidth(100);
        ImGui::SliderInt("Flag", &flag, 0, 4);
        ImGui::PopItemWidth();
        flag_ = flag;
    }

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("Clamp cotan", &clamp_cotan_);
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // turn mesh into non-triangles
    if (ImGui::CollapsingHeader("Polygons!", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Catmull-Clark subdivision
        if (ImGui::Button("Catmull-Clark"))
        {
            SurfaceSubdivision(mesh_).catmull_clark();
            update_mesh();
        }
        
        // dualize the mesh
        if (ImGui::Button("Dualize mesh"))
        {
            dualize();
        }

        if (ImGui::Button("Kugelize"))
        {
            for (auto v : mesh_.vertices())
                mesh_.position(v) = normalize(mesh_.position(v));
            update_mesh();
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // discrete harmonic parameterization
    if (ImGui::CollapsingHeader("Parametrization",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Discrete Harmonic"))
        {
            Parameterization(mesh_, flag_).harmonic_free_boundary();
            mesh_.use_checkerboard_texture();
            set_draw_mode("Texture");

            update_mesh();
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // implicit smoothing and energy smoothing
    if (ImGui::CollapsingHeader("Smoothing", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static float timestep = 0.1;
        float lb = 0.001;
        float ub = 1.0;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("TimeStep", &timestep, lb, ub);
        ImGui::PopItemWidth();
        if (ImGui::Button("Implicit Smoothing"))
        {
            close_holes();
            Scalar dt = timestep;
            smooth_.implicit_smoothing(dt);
            update_mesh();
            BoundingBox bb = mesh_.bounds();
            set_scene((vec3)bb.center(), 0.5 * bb.size());
            open_holes();
        }

        static float alpha = 0.01;
        float down = 0.001;
        float up = 1.0;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("Alpha", &alpha, down, up);
        ImGui::PopItemWidth();

        if (ImGui::Button("Energy Smoothing"))
        {
            close_holes();
            Scalar dt = alpha;
            energysmoothing_.energy_smoothing(dt);
            update_mesh();
            BoundingBox bb = mesh_.bounds();
            set_scene((vec3)bb.center(), 0.5 * bb.size());
            open_holes();
        }

        if (ImGui::Button("Boundary Smoothing"))
        {
            //close_holes();
            update_mesh();
            Scalar dt = alpha;
            BoundarySmoothing_.boundary_smoothing(dt);
            update_mesh();
            BoundingBox bb = mesh_.bounds();
            set_scene((vec3)bb.center(), 0.5 * bb.size());
            update_mesh();
            //open_holes();
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // curvature visualization
    if (ImGui::CollapsingHeader("Curvature", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Mean Curvature"))
        {
            Curvature curv(mesh_, flag_);
            curv.compute();
            curv.curvature_to_texture_coordinates();
            mesh_.use_cold_warm_texture();
            update_mesh();
            set_draw_mode("Texture");
        }
    }
    if (ImGui::CollapsingHeader("Geodesics in Heat",
                                ImGuiTreeNodeFlags_DefaultOpen))
    {
        static int  vertex_source = 0;
        ImGui::PushItemWidth(150);
        ImGui::InputInt("vertex_source", &vertex_source);
        ImGui::PopItemWidth();

        if (ImGui::Button("Compute Distances Vertex Source"))
        {
            int source = vertex_source;
            if (source < 0 || source >= mesh_.n_vertices())
            {
                exit(0);
            }
            GeodesicsInHeat heat(mesh_, flag_);
            heat.compute_distance_from(Vertex(source));
            heat.distance_to_texture_coordinates();
            mesh_.use_checkerboard_texture();
            //mesh_.use_cold_warm_texture();
            update_mesh();
            set_draw_mode("Texture");
        }
    }

    ImGui::Spacing();
    ImGui::Spacing();

    //deformation
    if (ImGui::CollapsingHeader("Deformation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        static float deltax = 0.0;
        float down = -0.1;
        float up = 0.1;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("DeltaX", &deltax, down, up);
        ImGui::PopItemWidth();

        static float deltay = 0.0;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("DeltaY", &deltay, down, up);
        ImGui::PopItemWidth();

        static float deltaz = 0.0;
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("DeltaZ", &deltaz, down, up);
        ImGui::PopItemWidth();

        if (ImGui::Button("Transform"))
        {
            close_holes();
            s.hPosition = Eigen::Vector3d(mesh_.position(Vertex(s.hIdx))) +
                          Eigen::Vector3d(deltax, deltay, deltaz);

            std::cout << "hPosition: " << s.hPosition << std::endl;
            deformation_.do_deform(s.hPosition, s.hIdx, s.nRoi, s.roiIndices);
            update_mesh();
            BoundingBox bb = mesh_.bounds();
            set_scene((vec3)bb.center(), 0.5 * bb.size());
            open_holes();
        }

        if (ImGui::Button("Reback"))
        {
            s.hPosition.setZero();
            s.hIdx = 0;
            s.nRoi = 0;
            s.roiIndices.clear();
            //mesh_.remove_vertex_property(s.colors);
            std::cout << "clear deformation data" << std::endl;
        }

        if (ImGui::Button("DoNoisy"))
        {
            close_holes();

            //pre_compute_normals
            pre_compute_normals(mesh_);
            
            std::default_random_engine e;
            std::uniform_real_distribution<double> u(-0.01, 0.01);
            for (auto v : mesh_.vertices())
            {
                mesh_.position(v) += Eigen::Vector3d(u(e), u(e), u(e));
            }
            std::cout << "do noisy operation" << std::endl;
            
            update_mesh();
            BoundingBox bb = mesh_.bounds();
            set_scene((vec3)bb.center(), 0.5 * bb.size());
            open_holes();

        }
    }
}

//----------------------------------------------------------------------------

void Viewer::draw(const std::string &draw_mode)
{
    // normal mesh draw
    mesh_.draw(projection_matrix_, modelview_matrix_, draw_mode);

    // draw uv layout
    if (mesh_.has_vertex_property("v:tex"))
    {
        // clear depth buffer
        glClear(GL_DEPTH_BUFFER_BIT);

        // setup viewport
        GLint size = std::min(width(), height()) / 4;
        glViewport(width() - size - 1, height() - size - 1, size, size);

        // setup matrices
        mat4 P = ortho_matrix(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);
        mat4 M = mat4::identity();

        // draw mesh once more
        mesh_.draw(P, M, "Texture Layout");

        // reset viewport
        glViewport(0, 0, width(), height());
    }
}

//----------------------------------------------------------------------------

void Viewer::dualize()
{
    SurfaceMeshGL dual;

    auto fvertex = mesh_.add_face_property<Vertex>("f:vertex");
    for (auto f : mesh_.faces())
    {
        fvertex[f] = dual.add_vertex(centroid(mesh_, f));
    }

    for (auto v : mesh_.vertices())
    {
        if (!mesh_.is_boundary(v))
        {
            std::vector<Vertex> vertices;
            for (auto f : mesh_.faces(v))
                vertices.push_back(fvertex[f]);
            dual.add_face(vertices);
        }
    }

    mesh_ = dual;
    update_mesh();
}

//----------------------------------------------------------------------------

void Viewer::update_mesh()
{
    // re-compute face and vertex normals
    mesh_.update_opengl_buffers();
}

//----------------------------------------------------------------------------

void Viewer::close_holes()
{
    bool finished = false;
    std::vector<Face> holes;
    while (!finished)
    {
        finished = true;

        // loop through all vertices
        for (auto v : mesh_.vertices())
        {
            // if we find a boundary vertex...
            if (mesh_.is_boundary(v))
            {
                // trace boundary loop
                std::vector<Vertex> vertices;
                vertices.push_back(v);
                for (Halfedge h = mesh_.halfedge(v); mesh_.to_vertex(h) != v;
                     h = mesh_.next_halfedge(h))
                {
                    vertices.push_back(mesh_.to_vertex(h));
                }

                // add boudary loop as polygonal face
                Face f = mesh_.add_face(vertices);
                holes.push_back(f);
                // start over
                finished = false;
                break;
            }
        }
    }
    holes_ = holes;
    update_mesh();
}

//----------------------------------------------------------------------------

void Viewer::open_holes()
{
    for (Face f : holes_)
        mesh_.delete_face(f);
    mesh_.garbage_collection();
    update_mesh();
}

//----------------------------------------------------------------------------

void Viewer::mouse(int button, int action, int mods)
{
    //mesh_.add_vertex_property<Color>("v:color");

    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_MIDDLE &&
        mods == GLFW_MOD_SHIFT)
    {
        double x, y;
        cursor_pos(x, y);
        Vertex v = pick_vertex(x, y);
        if (mesh_.is_valid(v))
        {
            GeodesicsInHeat heat(mesh_, flag_);
            heat.compute_distance_from(v);
            heat.distance_to_texture_coordinates();
            update_mesh();
            //mesh_.use_checkerboard_texture();
            mesh_.use_cold_warm_texture();
            set_draw_mode("Texture");
        }
    }
    else if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT &&
             mods == GLFW_MOD_SHIFT) // shift + left == select
    {
        double x, y;
        cursor_pos(x, y);
        Vertex v = pick_vertex(x, y);
        if (mesh_.is_valid(v))
        {
            s.hIdx = v.idx();
            //s.colors[v] = Eigen::Vector3d(220. / 255, 0. / 255, 102. / 255);
        }
        else
            return;
        std::cout << "hIdx: " << s.hIdx << std::endl;

        s.roiIndices.push_back(v.idx());
        s.nRoi++;
        /*
        for (Halfedge h = mesh_.halfedge(v); mesh_.to_vertex(h) != v && s.nRoi <= 100;
             h = mesh_.next_halfedge(h))
        {
            s.roiIndices.push_back((mesh_.to_vertex(h)).idx());
            s.nRoi++;
        }
        */
    }
    else if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT &&
             mods == GLFW_MOD_SHIFT) //shift + right == select roi
    {
        double x, y;
        cursor_pos(x, y);
        Vertex v = pick_vertex(x, y);
        if (mesh_.is_valid(v))
        {
            s.roiIndices.push_back(v.idx());
            s.nRoi++;
            //s.colors[v] = Eigen::Vector3d(30. / 255, 80. / 255, 255. / 255);
        }
        else
            return;

        std::cout << "ROI points is updating, count: " << s.nRoi << std::endl;
    }
    else if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT &&
             mods == GLFW_MOD_ALT) //alt + left ==  geodesics distance
    {
        double x, y;
        cursor_pos(x, y);
        Vertex v = pick_vertex(x, y);

        GeodesicsInHeat heat(mesh_, flag_);
        heat.compute_distance_from(Vertex(0));

        auto distances = mesh_.get_vertex_property<Scalar>("v:dist");
        assert(distances);

        // find maximum distance
        Scalar maxdist(0);
        for (auto v : mesh_.vertices())
        {
            if (distances[v] <= FLT_MAX)
            {
                maxdist = std::max(maxdist, distances[v]);
            }
        }
        std::cout << "maximun geodesics distance: " << maxdist<< std::endl;

        if (mesh_.is_valid(v))
        {
            std::cout << "geodesics distance: " << distances[v] << std::endl;
        }
        else
            return;
        
        std::cout << "relative geodesics distance: " << distances[v] / maxdist
                  << std::endl;
    }
    else
    {
        MeshViewer::mouse(button, action, mods);
    }
}

//=============================================================================
