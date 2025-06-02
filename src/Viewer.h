//=============================================================================
// Copyright 2020 Astrid Bunge, Philipp Herholz, Misha Kazhdan, Mario Botsch.
// Distributed under MIT license, see file LICENSE for details.
//=============================================================================
#pragma once
//=============================================================================

#include <pmp/visualization/MeshViewer.h>
#include "Smoothing.h"
#include "EnergySmoothing.h"
#include "BoundarySmoothing.h"
#include "Deformation.h"
//=============================================================================

class Viewer : public pmp::MeshViewer
{
public:
    Viewer(const char *title, int width, int height)
        : MeshViewer(title, width, height),
          smooth_(mesh_, flag_),
          energysmoothing_(mesh_, flag_),
          BoundarySmoothing_(mesh_, flag_),
          deformation_(mesh_, flag_)
    {
        set_draw_mode("Hidden Line");
        add_help_item("Shift + MMB", "Geod. distances for selected vertex", 5);
    }

    virtual void load_mesh(const char *filename) override;

protected:
    virtual void process_imgui() override;
    virtual void update_mesh() override;
    virtual void draw(const std::string &_draw_mode) override;
    virtual void mouse(int button, int action, int mods) override;

    void dualize();
    void close_holes();
    void open_holes();

private:
    BoundarySmoothing BoundarySmoothing_;
    EnergySmoothing energysmoothing_;
    Deformation deformation_;
    Smoothing smooth_;
    std::vector<Face> holes_;
    int flag_ = 2;
};

//=============================================================================
