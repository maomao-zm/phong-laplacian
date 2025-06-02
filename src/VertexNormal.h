#pragma once

#include "pmp/SurfaceMesh.h"

using namespace pmp;

class VertexNormals {
public:
    //ɾ��Ĭ�Ϻ͸��ƵĹ��캯��
    VertexNormals() = delete;
    VertexNormals(const VertexNormals&) = delete;
    
    //uniform weight
    static Normal compute_uniform_vertex_normal(const SurfaceMesh& mesh, Vertex v);
    //area weight
    static Normal compute_area_vertex_normal(const SurfaceMesh& mesh, Vertex v);
};