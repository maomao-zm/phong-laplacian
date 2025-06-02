// Copyright 2011-2022 the Polygon Mesh Processing Library developers.
// Distributed under a MIT-style license, see LICENSE.txt for details.

#include <pmp/visualization/MeshViewer.h>
#include <pmp/algorithms/SurfaceSubdivision.h>
#include <pmp/algorithms/SurfaceFeatures.h>
#include <pmp/algorithms/SurfaceTriangulation.h>
#include <imgui.h>

using namespace pmp;

class Viewer : public MeshViewer
{
public:
    Viewer(const char* title, int width, int height, bool showgui = true);

protected:
    virtual void process_imgui();
};

Viewer::Viewer(const char* title, int width, int height, bool showgui)
    : MeshViewer(title, width, height, showgui)
{
    set_draw_mode("Hidden Line");
    crease_angle_ = 0.0;
}

void Viewer::process_imgui()
{
    MeshViewer::process_imgui();

    ImGui::Spacing();
    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Subdivision", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (ImGui::Button("Triangulate Mesh"))
        {
            SurfaceTriangulation tesselator(mesh_);
            tesselator.triangulate();
            update_mesh();
        }

        if (ImGui::Button("Loop Subdivision"))
        {
            try
            {
                SurfaceSubdivision(mesh_).loop();
            }
            catch (const InvalidInputException& e)
            {
                std::cerr << e.what() << std::endl;
                return;
            }
            update_mesh();
        }

        if (ImGui::Button("Quad-Tri Subdivision"))
        {
            SurfaceSubdivision(mesh_).quad_tri();
            update_mesh();
        }

        if (ImGui::Button("Catmull-Clark Subdivision"))
        {
            SurfaceSubdivision(mesh_).catmull_clark();
            update_mesh();
        }

        if (ImGui::Button("Quad/Tri Subdivision"))
        {
            SurfaceSubdivision(mesh_).quad_tri();
            update_mesh();
        }
    }
}

int main(int argc, char** argv)
{
#ifndef __EMSCRIPTEN__
    Viewer window("Subdivision", 800, 600);
    if (argc == 2)
        window.load_mesh(argv[1]);
    return window.run();
#else
    Viewer window("Subdivision", 800, 600);
    window.load_mesh(argc == 2 ? argv[1] : "input.obj");
    return window.run();
#endif
}
