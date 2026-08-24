#include "utils.hpp"

#include <ftk/numeric/critical_point_type.hh>
#include <ftk/numeric/eigen_solver2.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/linear_interpolation.hh>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_map>

namespace {

struct CriticalPoint2D {
    int type;
};

struct RelativeErrorStats {
    double max_absolute_error;
    double original_range;
    double max_relative_eb;
};

RelativeErrorStats compute_max_relative_eb(
    const float* original, const float* decompressed, size_t count)
{
    double minimum = original[0];
    double maximum = original[0];
    double max_absolute_error = 0.0;

    for (size_t i = 0; i < count; ++i) {
        const double original_value = original[i];
        const double error = std::fabs(
            static_cast<double>(decompressed[i]) - original_value);
        if (original_value < minimum) minimum = original_value;
        if (original_value > maximum) maximum = original_value;
        if (error > max_absolute_error) max_absolute_error = error;
    }

    const double range = maximum - minimum;
    const double max_relative_eb = range > 0.0
        ? max_absolute_error / range
        : (max_absolute_error == 0.0
            ? 0.0 : std::numeric_limits<double>::infinity());
    return {max_absolute_error, range, max_relative_eb};
}

void check_simplex_for_cp(
    const float* u, const float* v,
    size_t i0, size_t i1, size_t i2,
    size_t simplex_id, const double coordinates[3][2],
    std::unordered_map<size_t, CriticalPoint2D>& critical_points)
{
    double vectors[3][2] = {
        {u[i0], v[i0]}, {u[i1], v[i1]}, {u[i2], v[i2]}
    };
    for (int k = 0; k < 3; k++) {
        if (vectors[k][0] == 0.0 && vectors[k][1] == 0.0) return;
    }

    double barycentric[3], condition_number;
    if (!ftk::inverse_lerp_s2v2(vectors, barycentric, &condition_number, 0.0)) return;

    double jacobian[2][2];
    ftk::jacobian_2dsimplex2(coordinates, vectors, jacobian);
    std::complex<double> eigenvalues[2];
    const double discriminant = ftk::solve_eigenvalues2x2(jacobian, eigenvalues);

    int type = ftk::CRITICAL_POINT_2D_UNKNOWN;
    if (discriminant >= 0.0) {
        if (eigenvalues[0].real() * eigenvalues[1].real() < 0.0) {
            type = ftk::CRITICAL_POINT_2D_SADDLE;
        }
        else if (eigenvalues[0].real() < 0.0 && eigenvalues[1].real() < 0.0) {
            type = ftk::CRITICAL_POINT_2D_ATTRACTING;
        }
        else if (eigenvalues[0].real() > 0.0 && eigenvalues[1].real() > 0.0) {
            type = ftk::CRITICAL_POINT_2D_REPELLING;
        }
    }
    else if (eigenvalues[0].real() < 0.0) {
        type = ftk::CRITICAL_POINT_2D_ATTRACTING_FOCUS;
    }
    else if (eigenvalues[0].real() > 0.0) {
        type = ftk::CRITICAL_POINT_2D_REPELLING_FOCUS;
    }
    else {
        type = ftk::CRITICAL_POINT_2D_CENTER;
    }

    if (type != ftk::CRITICAL_POINT_2D_UNKNOWN) {
        critical_points[simplex_id] = {type};
    }
}

std::unordered_map<size_t, CriticalPoint2D> compute_critical_points_2d(
    const float* u, const float* v, size_t r1, size_t r2)
{
    std::unordered_map<size_t, CriticalPoint2D> critical_points;
    const double lower_coordinates[3][2] = {{0, 0}, {0, 1}, {1, 1}};
    const double upper_coordinates[3][2] = {{0, 0}, {1, 0}, {1, 1}};

    // Keep the same interior-domain convention as the original verifier.
    for (size_t i = 1; i < r1 - 2; i++) {
        for (size_t j = 1; j < r2 - 2; j++) {
            const size_t simplex_base = 2 * (i * (r2 - 1) + j);
            const size_t a = i * r2 + j;
            const size_t b = (i + 1) * r2 + j;
            const size_t c = (i + 1) * r2 + (j + 1);
            const size_t d = i * r2 + (j + 1);
            check_simplex_for_cp(
                u, v, a, b, c, simplex_base, lower_coordinates, critical_points);
            check_simplex_for_cp(
                u, v, a, d, c, simplex_base + 1, upper_coordinates, critical_points);
        }
    }
    return critical_points;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 7) {
        fprintf(stderr,
            "Usage: %s original_U original_V decompressed_U decompressed_V r1 r2\n",
            argv[0]);
        return 1;
    }

    const size_t r1 = strtoull(argv[5], nullptr, 10);
    const size_t r2 = strtoull(argv[6], nullptr, 10);
    if (r1 < 4 || r2 < 4 || r1 > std::numeric_limits<size_t>::max() / r2) {
        fprintf(stderr, "Invalid dimensions: %s x %s\n", argv[5], argv[6]);
        return 1;
    }

    size_t original_u_count = 0, original_v_count = 0;
    size_t decompressed_u_count = 0, decompressed_v_count = 0;
    float* original_u = readfile<float>(argv[1], original_u_count);
    float* original_v = readfile<float>(argv[2], original_v_count);
    float* decompressed_u = readfile<float>(argv[3], decompressed_u_count);
    float* decompressed_v = readfile<float>(argv[4], decompressed_v_count);

    const size_t expected_count = r1 * r2;
    if (!original_u || !original_v || !decompressed_u || !decompressed_v ||
        original_u_count != expected_count || original_v_count != expected_count ||
        decompressed_u_count != expected_count || decompressed_v_count != expected_count) {
        fprintf(stderr,
            "Input size mismatch: expected %zu floats; got original U=%zu V=%zu, "
            "decompressed U=%zu V=%zu\n",
            expected_count, original_u_count, original_v_count,
            decompressed_u_count, decompressed_v_count);
        free(original_u); free(original_v);
        free(decompressed_u); free(decompressed_v);
        return 1;
    }

    const auto original_cp =
        compute_critical_points_2d(original_u, original_v, r1, r2);
    const auto decompressed_cp =
        compute_critical_points_2d(decompressed_u, decompressed_v, r1, r2);

    size_t matched = 0, false_positive = 0, false_negative = 0;
    size_t type_mismatch = 0;
    size_t reported_type_mismatch = 0;
    for (const auto& item : original_cp) {
        const auto found = decompressed_cp.find(item.first);
        if (found == decompressed_cp.end()) {
            false_negative++;
        }
        else if (found->second.type != item.second.type) {
            type_mismatch++;
            if (reported_type_mismatch < 5) {
                const size_t cell = item.first / 2;
                const size_t row = cell / (r2 - 1);
                const size_t col = cell % (r2 - 1);
                const bool upper = (item.first & 1u) != 0;
                const size_t i0 = row * r2 + col;
                const size_t i1 = upper ? i0 + 1 : i0 + r2;
                const size_t i2 = i0 + r2 + 1;
                printf(
                    "  type mismatch simplex=%zu cell=(%zu,%zu) tri=%s "
                    "type=%d->%d\n",
                    item.first, row, col, upper ? "upper" : "lower",
                    item.second.type, found->second.type);
                printf(
                    "    orig U/V: (% .9g,% .9g) (% .9g,% .9g) (% .9g,% .9g)\n",
                    original_u[i0], original_v[i0],
                    original_u[i1], original_v[i1],
                    original_u[i2], original_v[i2]);
                printf(
                    "    dec  U/V: (% .9g,% .9g) (% .9g,% .9g) (% .9g,% .9g)\n",
                    decompressed_u[i0], decompressed_v[i0],
                    decompressed_u[i1], decompressed_v[i1],
                    decompressed_u[i2], decompressed_v[i2]);
                reported_type_mismatch++;
            }
        }
        else {
            matched++;
        }
    }
    for (const auto& item : decompressed_cp) {
        if (original_cp.find(item.first) == original_cp.end()) false_positive++;
    }

    const bool passed =
        false_positive == 0 && false_negative == 0 && type_mismatch == 0;
    const auto u_error = compute_max_relative_eb(
        original_u, decompressed_u, expected_count);
    const auto v_error = compute_max_relative_eb(
        original_v, decompressed_v, expected_count);
    const double max_relative_eb = u_error.max_relative_eb > v_error.max_relative_eb
        ? u_error.max_relative_eb : v_error.max_relative_eb;

    printf("CP verification:\n");
    printf("  orig=%zu  decomp=%zu  matched=%zu  FP=%zu  FN=%zu  type_mismatch=%zu\n",
        original_cp.size(), decompressed_cp.size(), matched,
        false_positive, false_negative, type_mismatch);
    printf("  result: %s\n", passed ? "PASS" : "FAIL");
    printf("Max rel_eb (max absolute error / original range):\n");
    printf("  U: %.10e  (max_abs=%.10e, range=%.10e)\n",
        u_error.max_relative_eb, u_error.max_absolute_error, u_error.original_range);
    printf("  V: %.10e  (max_abs=%.10e, range=%.10e)\n",
        v_error.max_relative_eb, v_error.max_absolute_error, v_error.original_range);
    printf("  overall: %.10e\n", max_relative_eb);

    free(original_u); free(original_v);
    free(decompressed_u); free(decompressed_v);
    return passed ? 0 : 2;
}
