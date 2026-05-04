// basis_vectors.cpp
// Week 01 — Linear Algebra I | CUDA / C++ Track
// Assad's 52-Week NVIDIA Curriculum
//
// Demonstrates:
//   - Standard basis vectors e1, e2, e3 in R3
//   - Expressing any vector as a linear combination of basis vectors
//   - Verifying span: every R3 vector = c1*e1 + c2*e2 + c3*e3
//   - Connection to ML: token embeddings as linear combinations in Rd
//
// Compile: g++ -O2 -Wall -o basis_vectors basis_vectors.cpp
// Run:     ./basis_vectors

#include <iostream>
#include <cmath>    // fabs

// ─────────────────────────────────────────────────────────────
// Simple 3D vector struct (no STL — raw floats)
// ─────────────────────────────────────────────────────────────
struct Vec3 {
    float x, y, z;
};

// ─────────────────────────────────────────────────────────────
// Scalar multiply: scale * v
// ─────────────────────────────────────────────────────────────
Vec3 scalar_multiply(float scale, Vec3 v) {
    return {scale * v.x, scale * v.y, scale * v.z};
}

// ─────────────────────────────────────────────────────────────
// Vector addition
// ─────────────────────────────────────────────────────────────
Vec3 vec_add(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// ─────────────────────────────────────────────────────────────
// Reconstruct vector from linear combination of e1, e2, e3
// Any v in R3: v = v.x*e1 + v.y*e2 + v.z*e3
// ─────────────────────────────────────────────────────────────
Vec3 reconstruct_from_basis(Vec3 v, Vec3 e1, Vec3 e2, Vec3 e3) {
    Vec3 term1 = scalar_multiply(v.x, e1);
    Vec3 term2 = scalar_multiply(v.y, e2);
    Vec3 term3 = scalar_multiply(v.z, e3);
    return vec_add(vec_add(term1, term2), term3);
}

// ─────────────────────────────────────────────────────────────
// Print a Vec3
// ─────────────────────────────────────────────────────────────
void print_vec(const char* label, Vec3 v) {
    std::cout << "  " << label
              << " = [" << v.x << ", " << v.y << ", " << v.z << "]\n";
}

// ─────────────────────────────────────────────────────────────
// Verify two vectors are equal (within float tolerance)
// ─────────────────────────────────────────────────────────────
bool vec_equal(Vec3 a, Vec3 b, float tol = 1e-5f) {
    return fabs(a.x - b.x) < tol &&
           fabs(a.y - b.y) < tol &&
           fabs(a.z - b.z) < tol;
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
int main() {
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "Week 01 — Basis Vectors Demo\n";
    std::cout << "═══════════════════════════════════════════\n\n";

    // ── Standard basis for R3 ────────────────────────────────
    Vec3 e1 = {1.0f, 0.0f, 0.0f};
    Vec3 e2 = {0.0f, 1.0f, 0.0f};
    Vec3 e3 = {0.0f, 0.0f, 1.0f};

    std::cout << "Standard basis for R³:\n";
    print_vec("e1", e1);
    print_vec("e2", e2);
    print_vec("e3", e3);
    std::cout << "\n";

    // ── Linear independence check ─────────────────────────────
    // e1, e2, e3 are orthogonal → dot product between any two = 0
    float e1_e2 = e1.x*e2.x + e1.y*e2.y + e1.z*e2.z;
    float e1_e3 = e1.x*e3.x + e1.y*e3.y + e1.z*e3.z;
    float e2_e3 = e2.x*e3.x + e2.y*e3.y + e2.z*e3.z;
    std::cout << "Linear independence (dot products between basis vectors):\n";
    std::cout << "  e1 · e2 = " << e1_e2 << "  (0 → orthogonal)\n";
    std::cout << "  e1 · e3 = " << e1_e3 << "  (0 → orthogonal)\n";
    std::cout << "  e2 · e3 = " << e2_e3 << "  (0 → orthogonal)\n";
    std::cout << "  → Basis vectors are mutually orthogonal → linearly independent ✓\n\n";

    // ── Express vectors as linear combinations ────────────────
    Vec3 target1 = {3.0f, -2.0f, 5.0f};
    Vec3 target2 = {-1.0f, 7.0f, 0.0f};
    Vec3 target3 = {0.0f, 0.0f, 1.0f};  // e3 itself

    std::cout << "Expressing vectors as linear combinations of {e1, e2, e3}:\n\n";

    // target1
    Vec3 recon1 = reconstruct_from_basis(target1, e1, e2, e3);
    std::cout << "  v1 = [3, -2, 5]\n";
    std::cout << "     = 3·e1 + (-2)·e2 + 5·e3\n";
    std::cout << "     = 3·[1,0,0] + (-2)·[0,1,0] + 5·[0,0,1]\n";
    print_vec("     = reconstructed", recon1);
    std::cout << "     Match: " << (vec_equal(target1, recon1) ? "✓" : "✗") << "\n\n";

    // target2
    Vec3 recon2 = reconstruct_from_basis(target2, e1, e2, e3);
    std::cout << "  v2 = [-1, 7, 0]\n";
    std::cout << "     = (-1)·e1 + 7·e2 + 0·e3\n";
    print_vec("     = reconstructed", recon2);
    std::cout << "     Match: " << (vec_equal(target2, recon2) ? "✓" : "✗") << "\n\n";

    // target3
    Vec3 recon3 = reconstruct_from_basis(target3, e1, e2, e3);
    std::cout << "  v3 = [0, 0, 1]  (= e3 itself)\n";
    std::cout << "     = 0·e1 + 0·e2 + 1·e3\n";
    print_vec("     = reconstructed", recon3);
    std::cout << "     Match: " << (vec_equal(target3, recon3) ? "✓" : "✗") << "\n\n";

    // ── Higher-dimensional intuition (ML connection) ──────────
    std::cout << "═══════════════════════════════════════════\n";
    std::cout << "ML Connection — Why Basis Matters:\n\n";
    std::cout << "  A token embedding in GPT-2 lives in R^768.\n";
    std::cout << "  That means the embedding space has 768 basis dimensions.\n";
    std::cout << "  Every token is a linear combination of those 768 directions.\n";
    std::cout << "  The model LEARNS which combinations encode semantic meaning.\n\n";
    std::cout << "  'king'   ≈ a·e1 + b·e2 + ... + z·e768  (specific weights)\n";
    std::cout << "  'queen'  ≈ a'·e1 + b'·e2 + ... + z'·e768\n";
    std::cout << "  The directions in which 'king' and 'queen' are close\n";
    std::cout << "  encode royalty, power, etc. — geometry IS semantics.\n";
    std::cout << "═══════════════════════════════════════════\n";

    return 0;
}
