#include <stdio.h>
#include <stdlib.h>
#include "cgnslib.h"
#include <cuda_runtime.h>

// Useful macros for handling errors
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA Error: %s (error code %d)\n",            \
                    cudaGetErrorString(error), error);                     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Read full mesh info from a CGNS file
void readCGNSFile(const char *filename, float **x, float **y, float **z, int *n_nodes,
                  int **cell_node_indices, int *n_cells, int *n_cell_indices,
                  int **boundary_node_indices, int *n_boundaries, CGNS_ENUMT(ElementType_t) *elementType,
                  int **face_node_indices, int *n_faces, int *n_face_indices, CGNS_ENUMT(ElementType_t) *faceType) {
    int indexFile, indexBase, indexZone, nCoords, nSections, nBC;
    cgsize_t size[3], start, end;

    // Open the CGNS file
    if (cg_open(filename, CG_MODE_READ, &indexFile)) {
        fprintf(stderr, "Error opening CGNS file %s\n", filename);
        cg_error_exit();
    }

    indexBase = 1;
    indexZone = 1;

    // Read zone size
    cg_zone_read(indexFile, indexBase, indexZone, NULL, size);
    *n_nodes = size[0];
    *n_cells = size[1];

    // Allocate memory for node coordinates
    *x = (float *)malloc((*n_nodes) * sizeof(float));
    *y = (float *)malloc((*n_nodes) * sizeof(float));
    *z = (float *)malloc((*n_nodes) * sizeof(float));

    // Read node coordinates
    cg_ncoords(indexFile, indexBase, indexZone, &nCoords);
    cg_coord_read(indexFile, indexBase, indexZone, "CoordinateX", RealSingle, size, *x);
    cg_coord_read(indexFile, indexBase, indexZone, "CoordinateY", RealSingle, size, *y);
    cg_coord_read(indexFile, indexBase, indexZone, "CoordinateZ", RealSingle, size, *z);

    // Read sections for elements
    cg_nsections(indexFile, indexBase, indexZone, &nSections);
    for (int i = 1; i <= nSections; ++i) {
        cgsize_t numElements;

        cg_section_read(indexFile, indexBase, indexZone, i, NULL, elementType, &start, &end, 0, &numElements);

        if (*elementType == CGNS_ENUMV(TETRA_4)) {
            *n_cell_indices = (int)(numElements * 4); // 4 nodes per tetrahedron
            *cell_node_indices = (int *)malloc(*n_cell_indices * sizeof(int));
            cg_elements_read(indexFile, indexBase, indexZone, i, *cell_node_indices, NULL);
            break;
        } else if (*elementType == CGNS_ENUMV(HEXA_8)) {
            *n_cell_indices = (int)(numElements * 8); // 8 nodes per hexahedron
            *cell_node_indices = (int *)malloc(*n_cell_indices * sizeof(int));
            cg_elements_read(indexFile, indexBase, indexZone, i, *cell_node_indices, NULL);
            break;
        } else if (*elementType == CGNS_ENUMV(PRISM_6)) {
            *n_cell_indices = (int)(numElements * 6); // 6 nodes per prism
            *cell_node_indices = (int *)malloc(*n_cell_indices * sizeof(int));
            cg_elements_read(indexFile, indexBase, indexZone, i, *cell_node_indices, NULL);
            break;
        }
    }

    // Read face sections assuming optionally defined sections exist for faces
    for (int i = 1; i <= nSections; ++i) {
        cgsize_t numElements;
        cg_section_read(indexFile, indexBase, indexZone, i, NULL, faceType, &start, &end, 0, &numElements);

        if (*faceType == CGNS_ENUMV(QUAD_4) || *faceType == CGNS_ENUMV(TRI_3)) {
            *n_face_indices = (int)(numElements * (*faceType == CGNS_ENUMV(QUAD_4) ? 4 : 3));
            *face_node_indices = (int *)malloc(*n_face_indices * sizeof(int));
            cg_elements_read(indexFile, indexBase, indexZone, i, *face_node_indices, NULL);
            *n_faces = numElements;
            break; // Pick one section for simplicity
        }
    }

    // Read boundary conditions
    cg_nbocos(indexFile, indexBase, indexZone, &nBC);
    for (int i = 1; i <= nBC; ++i) {
        char bcName[33];
        CGNS_ENUMT(BCType_t) bcType;
        cgsize_t normalIndex[3];
        int nBCNodes, normalListFlag, ndataset;

        cg_boco_info(indexFile, indexBase, indexZone, i, bcName, &bcType, &nBCNodes, normalIndex, &normalListFlag, &ndataset);

        if (i == 1) { // Assume one boundary for simplicity; extend as needed
            *boundary_node_indices = (int *)malloc(nBCNodes * sizeof(int));
            cg_boco_read(indexFile, indexBase, indexZone, i, *boundary_node_indices, NULL);
            *n_boundaries = nBCNodes;
        }
    }

    // Close the CGNS file
    cg_close(indexFile);
}

// Helper function to cross product two 3D vectors
__device__ float3 crossProduct(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Helper function to compute the magnitude of a 3D vector
__device__ float magnitude(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// CUDA Kernel to process nodes, cells, faces, including surface normals and areas
__global__ void processMeshData(float *d_x, float *d_y, float *d_z,
                                int *d_cellNodeIndices, int *d_faceNodeIndices,
                                int *d_boundaryNodeIndices, int n_nodes, int n_cells,
                                int n_boundaries, int n_faces, CGNS_ENUMT(ElementType_t) faceType) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process nodes (simple example)
    if (idx < n_nodes) {
        d_x[idx] *= 2.0f;
        d_y[idx] *= 2.0f;
        d_z[idx] *= 2.0f;
    }

    // Process faces to compute normals and areas
    if (idx < n_faces) {
        float3 vertices[4]; // Supporting up to quad right now
        int nVertices = (faceType == CGNS_ENUMV(QUAD_4)) ? 4 : 3;

        // Load vertices based on face type
        for (int i = 0; i < nVertices; ++i) {
            int nodeIdx = d_faceNodeIndices[idx * nVertices + i];
            vertices[i] = make_float3(d_x[nodeIdx], d_y[nodeIdx], d_z[nodeIdx]);
        }
        
        // Use the first three vertices to form two vectors (common practice in triangular/quadrilateral geometries)
        float3 vec1 = make_float3(vertices[1].x - vertices[0].x, vertices[1].y - vertices[0].y, vertices[1].z - vertices[0].z);
        float3 vec2 = make_float3(vertices[2].x - vertices[0].x, vertices[2].y - vertices[0].y, vertices[2].z - vertices[0].z);

        // For quads, use the fourth vertex to correct area if needed:
        if (nVertices == 4) {
            float3 vec3 = make_float3(vertices[3].x - vertices[0].x, vertices[3].y - vertices[0].y, vertices[3].z - vertices[0].z);
            float3 normal1 = crossProduct(vec1, vec2);
            float3 normal2 = crossProduct(vec2, vec3);
            float3 combinedNormal = make_float3((normal1.x + normal2.x) * 0.5f, 
                                                (normal1.y + normal2.y) * 0.5f, 
                                                (normal1.z + normal2.z) * 0.5f);
            float area = (magnitude(normal1) + magnitude(normal2)) * 0.5f;
            printf("Face %d Normal: (%f, %f, %f), Area: %f\n", idx, combinedNormal.x, combinedNormal.y, combinedNormal.z, area);
        } else {
            // Otherwise, calculate normal and area for triangle
            float3 normal = crossProduct(vec1, vec2);
            float area = 0.5f * magnitude(normal);
            printf("Face %d Normal: (%f, %f, %f), Area: %f\n", idx, normal.x, normal.y, normal.z, area);
        }
    }

    // Print boundary node information
    if (idx < n_boundaries) {
        int bnIdx = d_boundaryNodeIndices[idx];
        printf("Boundary Node %d: (%f, %f, %f)\n", bnIdx, d_x[bnIdx], d_y[bnIdx], d_z[bnIdx]);
    }
}

int main() {
    const char *filename = "mesh.cgns";
    float *h_x, *h_y, *h_z;
    int *h_cellNodeIndices, *h_faceNodeIndices, *h_boundaryNodeIndices;
    float *d_x, *d_y, *d_z;
    int *d_cellNodeIndices, *d_faceNodeIndices, *d_boundaryNodeIndices;
    int n_nodes, n_cells, n_cell_indices, n_boundaries, n_faces, n_face_indices;
    CGNS_ENUMT(ElementType_t) elementType, faceType;

    // Read mesh data from CGNS file
    readCGNSFile(filename, &h_x, &h_y, &h_z, &n_nodes, &h_cellNodeIndices, &n_cells, &n_cell_indices, 
                 &h_boundaryNodeIndices, &n_boundaries, &elementType, &h_faceNodeIndices, &n_faces, &n_face_indices, &faceType);

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc((void **)&d_x, n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_z, n_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_cellNodeIndices, n_cell_indices * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_faceNodeIndices, n_face_indices * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_boundaryNodeIndices, n_boundaries * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, n_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, n_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_z, h_z, n_nodes * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellNodeIndices, h_cellNodeIndices, n_cell_indices * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_faceNodeIndices, h_faceNodeIndices, n_face_indices * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boundaryNodeIndices, h_boundaryNodeIndices, n_boundaries * sizeof(int), cudaMemcpyHostToDevice));

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_faces + threadsPerBlock - 1) / threadsPerBlock;

    // Launch CUDA kernel
    processMeshData<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, d_cellNodeIndices, d_faceNodeIndices, d_boundaryNodeIndices, 
                                                        n_nodes, n_cells, n_boundaries, n_faces, faceType);

    // Check for errors during kernel launch
    CUDA_CHECK(cudaGetLastError());

    // Clean up
    free(h_x);
    free(h_y);
    free(h_z);
    free(h_cellNodeIndices);
    free(h_faceNodeIndices);
    free(h_boundaryNodeIndices);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_cellNodeIndices));
    CUDA_CHECK(cudaFree(d_faceNodeIndices));
    CUDA_CHECK(cudaFree(d_boundaryNodeIndices));

    return 0;
}
