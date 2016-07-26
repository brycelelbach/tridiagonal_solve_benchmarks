////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Bryce Adelstein Lelbach aka wash
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

// Solves a one-dimensional diffusion equation using the implicit Backward Time,
// Centered Space (BTCS) finite difference method. The following problem is
// solve:
//
//     u_t = D * u_zz, z in (0, 1)
//
//     u(z, 0) = sin(N * pi * z)
//     u(0, t) = u(1, t) = 0
// 
// This problem has an exact solution:
//
//     u(z, t) = e ^ (-D * N^2 * pi^2 * t) * sin(N * pi * z)

#include <iomanip>
#include <algorithm>

#include <cmath>
#include <cfenv>
#include <cassert>
#include <cstdio>
#include <cstdint>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include "timers.hpp"
#include "get_env_variable.hpp"
#include "fp_utils.hpp"
#include "array3d.hpp"

using a_array_type = array3d<double, layout_left, ARRAYALIGN>;
using b_array_type = array3d<double, layout_left, ARRAYALIGN + 1*ARRAYPAD>;
using c_array_type = array3d<double, layout_left, ARRAYALIGN + 2*ARRAYPAD>;
using u_array_type = array3d<double, layout_left, ARRAYALIGN + 3*ARRAYPAD>;
using r_array_type = array3d<double, layout_left, ARRAYALIGN>;

///////////////////////////////////////////////////////////////////////////////
// dest = src

template <typename Size, std::size_t DestAlign, std::size_t SrcAlign>
inline void copy(
    Size j_begin
  , Size j_end
  , array3d<double, layout_left, DestAlign>& dest
  , array3d<double, layout_left, SrcAlign> const& src
    ) noexcept
{
    auto const nx = src.nx();
    auto const nz = src.nz();

    __assume(0 == (nx % 8)); 

    assert(dest.nx() == src.nx());
    assert(dest.ny() == src.ny());
    assert(dest.nz() == src.nz());

    for (int k = 0; k < nz; ++k)
        for (int j = j_begin; j < j_end; ++j)
        {
            double*       destp = dest(_, j, k);
            double const* srcp  = src (_, j, k);

            __assume_aligned(destp, 64);
            __assume_aligned(srcp, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
                destp[i] = srcp[i];
        }
}

template <std::size_t DestAlign, std::size_t SrcAlign>
inline void copy(
    array3d<double, layout_left, DestAlign>& dest
  , array3d<double, layout_left, SrcAlign> const& src
    ) noexcept
{
    using size_type =
        typename array3d<double, layout_left, SrcAlign>::size_type;
    copy(size_type(0), src.ny(), dest, src);
}

///////////////////////////////////////////////////////////////////////////////
// r = A*u - r

inline void residual(
    u_array_type::size_type j_begin
  , u_array_type::size_type j_end
  , r_array_type& r       // Residual
  , a_array_type const& a // Lower band
  , b_array_type const& b // Diagonal
  , c_array_type const& c // Upper band
  , u_array_type const& u // Solution
    ) noexcept
{
    auto const nx = r.nx();
    auto const nz = r.nz();

    __assume(0 == (nx % 8)); 

    assert(r.nx()     == a.nx());
    assert(r.ny()     == a.ny());
    assert(r.nz()     == a.nz());

    assert(r.nx()     == b.nx());
    assert(r.ny()     == b.ny());
    assert(r.nz()     == b.nz());

    assert(r.nx()     == c.nx());
    assert(r.ny()     == c.ny());
    assert(r.nz()     == c.nz());

    assert(r.nx()     == u.nx());
    assert(r.ny()     == u.ny());
    assert(r.nz()     == u.nz());

    // First row.
    for (int j = j_begin; j < j_end; ++j)
    {
        double*       r0p = r(_, j, 0);

        double const* b0p = b(_, j, 0);

        double const* c0p = c(_, j, 0);

        double const* u0p = u(_, j, 0);
        double const* u1p = u(_, j, 1);

        __assume_aligned(r0p, 64);

        __assume_aligned(b0p, 64);

        __assume_aligned(c0p, 64);

        __assume_aligned(u0p, 64);
        __assume_aligned(u1p, 64);

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed.
            // r[0] = (b[0] * u[0] + c[0] * u[1]) - r[0];
            r0p[i] = (b0p[i] * u0p[i] + c0p[i] * u1p[i]) - r0p[i];
        }
    }

    // Interior rows.
    for (int k = 1; k < nz - 1; ++k)
        for (int j = j_begin; j < j_end; ++j)
        {
            double*       rp     = r(_, j, k);

            double const* asub1p = a(_, j, k - 1);

            double const* bp     = b(_, j, k);

            double const* cp     = c(_, j, k);

            double const* usub1p = u(_, j, k - 1);
            double const* up     = u(_, j, k);
            double const* uadd1p = u(_, j, k + 1);

            __assume_aligned(rp, 64);

            __assume_aligned(asub1p, 64);

            __assume_aligned(bp, 64);

            __assume_aligned(cp, 64);

            __assume_aligned(usub1p, 64);
            __assume_aligned(up, 64);
            __assume_aligned(uadd1p, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                // r[k] = ( a[k - 1] * u[k - 1]
                //        + b[k] * u[k]
                //        + c[k] * u[k + 1])
                //      - r[k];
                rp[i] = ( asub1p[i] * usub1p[i]
                        + bp[i] * up[i]
                        + cp[i] * uadd1p[i])
                      - rp[i];
            }
        }

    // Last row.
    for (int j = j_begin; j < j_end; ++j)
    {
        double*       rnz1p = r(_, j, nz - 1);

        double const* anz2p = a(_, j, nz - 2);

        double const* bnz1p = b(_, j, nz - 1);

        double const* unz2p = u(_, j, nz - 2);
        double const* unz1p = u(_, j, nz - 1);

        __assume_aligned(rnz1p, 64);

        __assume_aligned(anz2p, 64);

        __assume_aligned(bnz1p, 64);

        __assume_aligned(unz2p, 64);
        __assume_aligned(unz1p, 64);

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed.
            // r[nz - 1] = (a[nz - 2] * u[nz - 2] + b[nz - 1] * u[nz - 1])
            //           - r[nz - 1];
            rnz1p[i] = (anz2p[i] * unz2p[i] + bnz1p[i] * unz1p[i])
                     - rnz1p[i];
        }
    }
}

inline void residual(
    r_array_type& r       // Residual
  , a_array_type const& a // Lower band
  , b_array_type const& b // Diagonal
  , c_array_type const& c // Upper band
  , u_array_type const& u // Solution
    ) noexcept
{
    residual(0, r.ny(), r, a, b, c, u);
}

///////////////////////////////////////////////////////////////////////////////
// A*x = u; u = x

template <std::uint64_t NRIterations = 1>
inline double newton_raphson_divide(
    double num, double den
    ) noexcept __attribute__((always_inline));

/// Sandybridge:
///
///    P0   P1
///    RCP
/// /> MUL  ADD
/// |  MUL
/// \-      ADD
///    MUL
///
/// RCP + NRIterations * (2 * MUL + ADD) + MUL [FLOPS per point]
///   1 + NRIterations * (2 *   1 +   1) +   1 [FLOPS per point]
/// 
/// THEORETICAL: 2 + 3 * NRIterations [FLOPS per point]
///
/// Haswell:
///
///    P0   P1
///    RCP
/// /> FMA
/// \- FMA
///    MUL
///
/// RCP + NRIterations * (2 * FMA) + MUL [FLOPS per point]
///   1 + NRIterations * (2 *   1) +   1 [FLOPS per point]
/// 
/// THEORETICAL: 2 + 2 * NRIterations [FLOPS per point]
template <std::uint64_t NRIterations>
inline double newton_raphson_divide(double num, double den) noexcept
{
    /// RCP
    double x = float(1.0) / float(den);

    #pragma unroll
    for (int i = 0; i < NRIterations; ++i)
        /// Sandybridge:
        /// t0 = den * x;          MUL 
        /// t1 = x * t0;           MUL (dep t0)
        /// t2 = x + x;            ADD 
        /// x  = t2 - t1;          ADD (dep t2, t1)
        /// 
        /// Haswell:
        /// t0 = - den * x + 1.0;  FMA 
        /// x  = x * t0 + x;       FMA (dep t0)
        x = x + x * (1.0 - den * x);

    /// MUL (dep x)
    return num * x;
}

inline void tridiagonal_solve_native(
    u_array_type::size_type j_begin
  , u_array_type::size_type j_end
  , a_array_type& a    // Lower band
  , b_array_type& b    // Diagonal
  , c_array_type& c    // Upper band
  , u_array_type& u    // Solution
    ) noexcept
{
    auto const nx = u.nx();
    auto const nz = u.nz();

    __assume(0 == (nx % 8)); 

    assert(u.nx()     == a.nx());
    assert(u.ny()     == a.ny());
    assert(u.nz()     == a.nz());

    assert(u.nx()     == b.nx());
    assert(u.ny()     == b.ny());
    assert(u.nz()     == b.nz());

    assert(u.nx()     == c.nx());
    assert(u.ny()     == c.ny());
    assert(u.nz()     == c.nz());

    // Pre-Elimination: (j_end - j_begin) * (nx) iterations
    /// TODO
    for (int j = j_begin; j < j_end; ++j)
    {
        double* bbeginp = b(_, j, 0);

        __assume_aligned(bbeginp, 64);

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // b^-1[0] = 1.0 / b[0] 
            bbeginp[i] = newton_raphson_divide(1.0, bbeginp[i]);
        }
    }

    // Forward Elimination: (nz - 1) * (j_end - j_begin) * (nx) iterations
    /// TODO
    for (int k = 1; k < nz; ++k)
        for (int j = j_begin; j < j_end; ++j)
        {
            double* asub1p = a(_, j, k - 1);

            double* bp     = b(_, j, k);
            double* bsub1p = b(_, j, k - 1);

            double* csub1p = c(_, j, k - 1);

            double* up     = u(_, j, k);
            double* usub1p = u(_, j, k - 1);

            __assume_aligned(asub1p, 64);

            __assume_aligned(bp, 64);
            __assume_aligned(bsub1p, 64);

            __assume_aligned(csub1p, 64);

            __assume_aligned(up, 64);
            __assume_aligned(usub1p, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                /// TODO
                
                // double const m0 = a[k - 1] * b^-1[k - 1];
                // b[k] = 1.0 / (b[k] - m0 * c[k - 1]);
                // u[k] = u[k] - m0 * u[k - 1];
                using cdouble = double const;
                cdouble m0 = asub1p[i] * bsub1p[i];
                cdouble b0 = newton_raphson_divide(1.0, bp[i] - m0 * csub1p[i]);
                up[i] -= m0 * usub1p[i];
                bp[i] = b0; 
            }
        }

    // Pre-Substitution: (j_end - j_begin) * (nx) iterations
    /// TODO
    for (int j = j_begin; j < j_end; ++j)
    {
        double* bendp = b(_, j, nz - 1);

        double* uendp = u(_, j, nz - 1);

        __assume_aligned(bendp, 64);

        __assume_aligned(uendp, 64);

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // u[nz - 1] = u[nz - 1] * b^-1[nz - 1];
            uendp[i] *= bendp[i];
        }
    }
 
    // Back Substitution: (nz - 1) * (j_end - j_begin) * (nx) iterations
    /// TODO
    for (int k = nz - 2; k >= 0; --k)
        for (int j = j_begin; j < j_end; ++j)
        {
            double* bp     = b(_, j, k);

            double* cp     = c(_, j, k);

            double* up     = u(_, j, k);
            double* uadd1p = u(_, j, k + 1);

            __assume_aligned(bp, 64);

            __assume_aligned(cp, 64);

            __assume_aligned(up, 64);
            __assume_aligned(uadd1p, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                /// TODO

                // u[k] = (u[k] - c[k] * u[k + 1]) * b^-1[k];
                up[i] = (up[i] - cp[i] * uadd1p[i]) * bp[i];
            }
        }
}

///////////////////////////////////////////////////////////////////////////////

struct heat_equation_btcs
{
    using timer = high_resolution_timer;

    bool verify;
    bool header;

    double D;

    std::uint64_t nx;
    std::uint64_t ny;
    std::uint64_t nz;
    std::uint64_t tw;

    std::uint64_t ns;
    double dt;

  private:
    double dz;

    static double constexpr N = 1.0;

    double A_coef;

    a_array_type a;
    b_array_type b;
    c_array_type c;

    u_array_type u;

    r_array_type r;

    #if defined(_OPENMP)
        std::vector<timer::value_type> solvertimes;
    #endif

  public:
    heat_equation_btcs() noexcept
    {
        verify = get_env_variable<bool>("verify", false);
        header = get_env_variable<bool>("header", false);

        D  = get_env_variable<double>("D", 0.1);

        nx = get_env_variable<std::uint64_t>("nx", 32);
        ny = get_env_variable<std::uint64_t>("ny", 2048);
        nz = get_env_variable<std::uint64_t>("nz", 32);
        tw = get_env_variable<std::uint64_t>("tw", 16);

        ns = get_env_variable<std::uint64_t>("ns", 50);
        dt = get_env_variable<double>("dt", 1.0e-7);

        assert(0 == (nx % 8)); // Ensure 64-byte alignment.
        assert(8 <= nx); 

        assert(0 == (ny % tw));
    }

    void initialize() noexcept 
    { 
        // Compute grid spacing.
        dz = 1.0 / (nz - 1);

        // Compute matrix constant.
        A_coef = D * dt / (dz * dz);

        // Allocate storage for the matrix. a and c technically have dimensions
        // of nx * ny * (nz - 1), but we make then nx * ny * nz to simplify the
        // padding math.
        a.resize(nx, ny, nz, 0, NYPAD, 0);
        b.resize(nx, ny, nz, 0, NYPAD, 0);
        c.resize(nx, ny, nz, 0, NYPAD, 0);

        // Allocate storage for the problem state and initialize it.
        u.resize(nx, ny, nz, 0, NYPAD, 0);

        __assume(0 == (nx % 8));

        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            {
                double* up = u(_, j, k);

                __assume_aligned(up, 64);

                #pragma simd
                for (int i = 0; i < nx; ++i)
                    up[i] = std::sin(N * M_PI * (dz * k));
            }

        // Allocate storage for the residual.
        r.resize(nx, ny, nz);

        #if defined(_OPENMP)
            // Allocate storage for the per-thread solver walltime count.
            solvertimes.resize(omp_get_max_threads(), 0.0);
        #endif
    }

    void build_matrix(std::ptrdiff_t j_begin, std::ptrdiff_t j_end) noexcept
    {
        double const ac_term = -A_coef;
        double const b_term  = 1.0 + 2.0 * A_coef;

        __assume(0 == (nx % 8)); 
 
        for (int k = 0; k < nz; ++k)
            for (int j = j_begin; j < j_end; ++j)
            {
                double* bp = b(_, j, k);

                __assume_aligned(bp, 64);

                #pragma simd
                for (int i = 0; i < nx; ++i)
                    bp[i] = b_term;
            }

        for (int k = 0; k < nz - 1; ++k)
            for (int j = j_begin; j < j_end; ++j)
            {
                double* ap = a(_, j, k);
                double* cp = c(_, j, k);

                __assume_aligned(ap, 64);
                __assume_aligned(cp, 64);

                #pragma simd
                for (int i = 0; i < nx; ++i)
                {
                    ap[i] = ac_term;
                    cp[i] = ac_term;
                }
            }

        // Boundary conditions.
        for (int j = j_begin; j < j_end; ++j)
        {
            double* bbeginp = b(_, j, 0);
            double* cbeginp = c(_, j, 0);

            double* aendp   = a(_, j, nz - 2);
            double* bendp   = b(_, j, nz - 1);

            __assume_aligned(bbeginp, 64);
            __assume_aligned(cbeginp, 64);

            __assume_aligned(aendp, 64);
            __assume_aligned(bendp, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                bbeginp[i] = 1.0; cbeginp[i] = 0.0;
                aendp[i]   = 0.0; bendp[i]   = 1.0;
            }
        }
    }

    double l2_norm(std::uint64_t step) noexcept
    {
        double l2 = 0.0;

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                double sum = 0.0;

                double const* up = u(i, j, _);

                auto const stride = u.stride_z();

                __assume_aligned(up, 64);

                // NOTE: Strided access.
                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    auto const ks = k * stride;

                    double const exact = std::exp( -D * (N * N)
                                                 * (M_PI * M_PI) * (dt * step))
                                       * std::sin(N * M_PI * (dz * k)); 

                    double const abs_term = std::fabs(up[ks] - exact);
                    sum = sum + abs_term * abs_term;
                }

                double const l2_here = std::sqrt(sum);

                if ((0 == i) && (0 == j))
                    // First iteration, so we have nothing to compare against.
                    l2 = l2_here;
                else
                    // All the columns are the same, so the L2 norm for each
                    // column should be the same.
                    assert(fp_equals(l2, l2_here));
            }

        return l2;
    }

    double max_residual() noexcept
    {
        residual(r, a, b, c, u);

        double mr = 0.0; 

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                double min = 1.0e300;
                double max = -1.0e300;

                double const* rp = r(i, j, _);

                auto const stride = r.stride_z();

                __assume_aligned(rp, 64);

                // NOTE: Strided access.
                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    auto const ks = k * stride;

                    min = std::min(min, rp[ks]);
                    max = std::max(max, rp[ks]);
                }

                double const mr_here = std::max(std::fabs(min), std::fabs(max));

                if ((0 == i) && (0 == j))
                    // First iteration, so we have nothing to compare against.
                    mr = mr_here;
                else
                    // All the columns are the same, so the max residual for
                    // each column should be the same.
                    assert(fp_equals(mr, mr_here));
            }

        return mr;
    }

    void solve()
    {
        initialize();

        timer t;

        #if !defined(_OPENMP)
            timer::value_type solvertime = 0.0;
        #endif

        for (int s = 0; s < ns; ++s)
        {
            if (verify)
                copy(r, u);

            #pragma omp parallel for schedule(static) 
            for (int j = 0; j < ny; j += tw)
            {
                std::ptrdiff_t const j_begin = j;
                std::ptrdiff_t const j_end   = j + tw;

                build_matrix(j_begin, j_end);
            }

            #pragma omp parallel for schedule(static)
            for (int j = 0; j < ny; j += tw)
            {
                std::ptrdiff_t const j_begin = j;
                std::ptrdiff_t const j_end   = j + tw;

                timer st;

                tridiagonal_solve_native(j_begin, j_end, a, b, c, u);

                #if defined(_OPENMP)
                    solvertimes[omp_get_thread_num()] += st.elapsed();
                #else
                    solvertime += st.elapsed();
                #endif
            }

            if (verify)
            {
                for (int j = 0; j < ny; j += tw)
                {
                    std::ptrdiff_t const j_begin = j;
                    std::ptrdiff_t const j_end   = j + tw;

                    build_matrix(j_begin, j_end);
                }

                double const resid = max_residual();

                double const l2 = l2_norm(s + 1);  

                std::printf(
                    "STEP %04u : "
                    "TIME %-10.7g = %-10.7g + %-10.7g : "
                    "L2 NORM %-22.17g : "
                    "MAX RESIDUAL %-22.17g\n"
                  , s
                  , dt * (s + 1)
                  , dt * s
                  , dt
                  , l2
                  , resid
                );
            }
        }

        timer::value_type const walltime = t.elapsed();

        #if defined(_OPENMP)
            timer::value_type solvertime = 0.0;

            for (timer::value_type t : solvertimes)
                solvertime += t;

            solvertime /= timer::value_type(omp_get_max_threads());
        #endif

        double const l2 = l2_norm(ns);  

        if (header)
            std::cout <<
                "Algorithm,"
                "Build Type,"
                "Diffusion Coefficient (D),"
                "X (Horizontal) Extent (nx),"
                "Y (Horizontal) Extent (ny),"
                "Z (Vertical) Extent (nz),"
                "Tile Width (tw),"
                "# of Timesteps (ns),"
                "Timestep Size (dt),"
                "# of Threads,"
                "Wall Time " << t.units() << ","
                "Solver Time " << t.units() << ","
                "L2 Norm"
                ;

        std::cout
            << "Streaming Mixed Precision NR Stored RCP,"
            << BUILD_TYPE << ","
            << D << ","
            << nx << ","
            << ny << ","
            << nz << ","
            << tw << ","
            << ns << ","
            << dt << ","
            #if defined(_OPENMP)
                << omp_get_max_threads() << ","
            #else
                << 1 << ","
            #endif
            << std::setprecision(7) << walltime << ","
            << std::setprecision(7) << solvertime << ","
            << std::setprecision(17) << l2 << "\n"
            ;
    }
};

int main()
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    heat_equation_btcs s;

    s.solve();
}

