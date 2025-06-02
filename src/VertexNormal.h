#pragma once

#include "pmp/SurfaceMesh.h"

using namespace pmp;

class VertexNormals {
public:
    //删除默认和复制的构造函数
    VertexNormals() = delete;
    VertexNormals(const VertexNormals&) = delete;
    
    //uniform weight
    static Normal compute_uniform_vertex_normal(const SurfaceMesh& mesh, Vertex v);
    //area weight
    static Normal compute_area_vertex_normal(const SurfaceMesh& mesh, Vertex v);
};