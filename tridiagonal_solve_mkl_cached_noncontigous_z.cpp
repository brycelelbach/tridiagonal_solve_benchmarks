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

#include <mkl.h>

#include "high_resolution_timer.hpp"
#include "get_env_variable.hpp"
#include "fp_utils.hpp"
#include "array3d.hpp"

///////////////////////////////////////////////////////////////////////////////
// dest = src

inline void copy(
    array3d<double, layout_left>::size_type j_begin
  , array3d<double, layout_left>::size_type j_end
  , array3d<double, layout_left>& dest
  , array3d<double, layout_left> const& src
    ) noexcept
{
    array3d<double, layout_left>::size_type const nx = src.nx();
    array3d<double, layout_left>::size_type const nz = src.nz();

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

inline void copy(
    array3d<double, layout_left>& dest
  , array3d<double, layout_left> const& src
    ) noexcept
{
    copy(0, src.ny(), dest, src);
}

///////////////////////////////////////////////////////////////////////////////
// r = A*u - r

inline void residual(
    array3d<double, layout_left>::size_type j_begin
  , array3d<double, layout_left>::size_type j_end
  , array3d<double, layout_left>& r                  // Residual
  , decltype(make_aligned_array<double>(0)) const& a // Lower band
  , decltype(make_aligned_array<double>(0)) const& b // Diagonal 
  , decltype(make_aligned_array<double>(0)) const& c // Upper band 
  , array3d<double, layout_left> const& u            // Solution
    ) noexcept
{
    array3d<double, layout_left>::size_type const nx = r.nx();
    array3d<double, layout_left>::size_type const nz = r.nz();

    __assume(0 == (nx % 8)); 
    __assume(0 == (nz % 8)); 

    assert(r.nx() == u.nx());
    assert(r.ny() == u.ny());
    assert(r.nz() == u.nz());

    // First row.
    for (int j = j_begin; j < j_end; ++j)
    {
        double*       r0p = r(_, j, 0);

        double const* u0p = u(_, j, 0);
        double const* u1p = u(_, j, 1);

        __assume_aligned(r0p, 64);

        __assume_aligned(u0p, 64);
        __assume_aligned(u1p, 64);

        double const b0 = b[0];

        double const c0 = c[0];

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed for
            // accesses to u[] and r[].
            // r[0] = (b[0] * u[0] + c[0] * u[1]) - r[0];
            r0p[i] = (b0 * u0p[i] + c0 * u1p[i]) - r0p[i];
        }
    }

    // Interior rows.
    for (int k = 1; k < nz - 1; ++k)
        for (int j = j_begin; j < j_end; ++j)
        {
            double*       rp     = r(_, j, k);

            double const* ap = a.get();

            double const* bp = b.get();

            double const* cp = c.get();

            double const* usub1p = u(_, j, k - 1);
            double const* up     = u(_, j, k);
            double const* uadd1p = u(_, j, k + 1);

            __assume_aligned(rp, 64);

            __assume_aligned(ap, 64);

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
                rp[i] = ( ap[k - 1] * usub1p[i]
                        + bp[k] * up[i]
                        + cp[k] * uadd1p[i])
                      - rp[i];
            }
        }

    // Last row.
    for (int j = j_begin; j < j_end; ++j)
    {
        double*       rnz1p = r(_, j, nz - 1);

        double const* unz2p = u(_, j, nz - 2);
        double const* unz1p = u(_, j, nz - 1);

        __assume_aligned(rnz1p, 64);

        __assume_aligned(unz2p, 64);
        __assume_aligned(unz1p, 64);

        double const anz2 = a[nz - 2];

        double const bnz1 = b[nz - 1];

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed for
            // accesses to u[] and r[].
            // r[nz - 1] = (a[nz - 2] * u[nz - 2] + b[nz - 1] * u[nz - 1])
            //           - r[nz - 1];
            rnz1p[i] = (anz2 * unz2p[i] + bnz1 * unz1p[i])
                     - rnz1p[i];
        }
    }
}

inline void residual(
    array3d<double, layout_left>& r                  // Residual
  , decltype(make_aligned_array<double>(0)) const& a // Lower band
  , decltype(make_aligned_array<double>(0)) const& b // Diagonal 
  , decltype(make_aligned_array<double>(0)) const& c // Upper band 
  , array3d<double, layout_left> const& u            // Solution
    ) noexcept
{
    residual(0, r.ny(), r, a, b, c, u);
}

///////////////////////////////////////////////////////////////////////////////

struct heat_equation_btcs
{
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

    decltype(make_aligned_array<double>(0)) a;
    decltype(make_aligned_array<double>(0)) b;
    decltype(make_aligned_array<double>(0)) c;

    array3d<double, layout_left> u;

    array3d<double, layout_left> r;

    decltype(make_aligned_array<double>(0)) u_buf;

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

        assert(0 == (nz % 8)); // Ensure 64-byte alignment.
        assert(8 <= nz); 

        assert(0 == (ny % tw));
    }

    void initialize() noexcept 
    { 
        // Compute grid spacing.
        dz = 1.0 / (nz - 1);

        // Compute matrix constant.
        A_coef = D * dt / (dz * dz);

        // Allocate storage for the matrix.
        a = make_aligned_array<double>(nz - 1);
        b = make_aligned_array<double>(nz);
        c = make_aligned_array<double>(nz - 1);

        // Allocate storage for the problem state and initialize it.
        u.resize(nx, ny, nz);

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

        // Allocate storage for gather buffers.
        u_buf = make_aligned_array<double>(nz);
    }

    void build_matrix() noexcept
    {
        double const ac_term = -A_coef;
        double const b_term  = 1.0 + 2.0 * A_coef;

        __assume(0 == (nz % 8)); 

        double* bp = b.get();

        __assume_aligned(bp, 64);

        #pragma simd
        for (int k = 0; k < nz; ++k)
            bp[k] = b_term;

        double* ap = a.get();
        double* cp = c.get();

        __assume_aligned(ap, 64);
        __assume_aligned(cp, 64);

        #pragma simd
        for (int k = 0; k < nz - 1; ++k)
        {
            ap[k] = ac_term;
            cp[k] = ac_term;
        }

        // Boundary conditions.
        b[0]      = 1.0; c[0]      = 0.0;
        a[nz - 2] = 0.0; b[nz - 1] = 1.0;
    }

    double l2_norm(std::uint64_t step) noexcept
    {
        double l2 = 0.0;

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                double sum = 0.0;

                double const* up = u(i, j, _);

                array3d<double, layout_left>::size_type const
                    stride = u.stride_z();

                __assume_aligned(up, 64);

                // NOTE: Strided access.
                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    array3d<double, layout_left>::size_type const
                        ks = k * stride;

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

                array3d<double, layout_left>::size_type const
                    stride = r.stride_z();

                __assume_aligned(rp, 64);

                // NOTE: Strided access.
                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    array3d<double, layout_left>::size_type const
                        ks = k * stride;

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

        high_resolution_timer t;

        double solvertime = 0.0;

        for (int s = 0; s < ns; ++s)
        {
            if (verify)
                copy(r, u);

            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    build_matrix();

                    high_resolution_timer st;

                    array3d<double, layout_left>::size_type const
                        stride = u.stride_z();

                    double* up = u(i, j, _); 

                    double* u_bufp = u_buf.get();

                    __assume_aligned(up, 64);

                    __assume_aligned(u_bufp, 64);

                    // NOTE: Strided access.
                    #pragma simd
                    for (int k = 0; k < nz; ++k)
                    {
                        array3d<double, layout_left>::size_type const
                            ks = k * stride;

                        u_bufp[k] = up[ks];
                    }

                    int mkl_n    = nz;
                    int mkl_nrhs = 1;
                    int mkl_ldb  = nz;
                    int mkl_info = 0;

                    dgtsv_(
                        &mkl_n,       // Matrix order.
                        &mkl_nrhs,    // # of right hand sides.
                        a.get(),      // Subdiagonal part.
                        b.get(),      // Diagonal part.
                        c.get(),      // Superdiagonal part.
                        u_buf.get(),  // Column to solve.
                        &mkl_ldb,     // Leading dimension of RHS.
                        &mkl_info
                        );

                    assert(mkl_info == 0);

                    // NOTE: Strided access.
                    #pragma simd
                    for (int k = 0; k < nz; ++k)
                    {
                        array3d<double, layout_left>::size_type const
                            ks = k * stride;

                        up[ks] = u_bufp[k];
                    }

                    solvertime += st.elapsed();
                }

            if (verify)
            {
                build_matrix();

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

        double const walltime = t.elapsed();

        double const l2 = l2_norm(ns);  

        if (header)
            std::cout <<
                "Algorithm,"
                "Build Type,"
                "Diffusion Coefficient (D),"
                "X (Horizontal) Extent (nx),"
                "Y (Horizontal) Extent (ny),"
                "Z (Horizontal) Extent (nz),"
                "Tile Width (tw),"
                "# of Timesteps (ns),"
                "Timestep Size (dt),"
                "Wall Time [s],"
                "Solver Time [s],"
                "L2 Norm"
                ;

        std::cout
            << "MKL Cached Noncontigous Z,"
            << BUILD_TYPE << ","
            << D << ","
            << nx << ","
            << ny << ","
            << nz << ","
            << tw << ","
            << ns << ","
            << dt << ","
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

