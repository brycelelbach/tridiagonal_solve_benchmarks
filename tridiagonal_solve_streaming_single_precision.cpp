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

#include "high_resolution_timer.hpp"
#include "get_env_variable.hpp"
#include "fp_utils.hpp"

///////////////////////////////////////////////////////////////////////////////

struct placeholder {};

constexpr placeholder _ {};

template <typename T, std::uint64_t Alignment = 64>
struct array3d
{
    typedef std::ptrdiff_t size_type;
    typedef T value_type;

  private:
    T* data_;
    size_type nx_, ny_, nz_;

  public:
    constexpr array3d() noexcept : data_(), nx_(), ny_(), nz_() {}

    array3d(size_type nx, size_type ny, size_type nz) noexcept
    {
        resize(nx, ny, nz);
    }

    ~array3d()
    {
        clear();
    }

    void resize(size_type nx, size_type ny, size_type nz) noexcept
    {
        clear();

        assert(0 == ((nx * ny * nz * sizeof(T)) % Alignment));

        void* p = 0; 
        int const r = posix_memalign(&p, Alignment, nx * ny * nz * sizeof(T));
        assert(0 == r);

        std::memset(p, 0, nx * ny * nz * sizeof(T));

        data_ = reinterpret_cast<T*>(p);

        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
    }

    void clear() noexcept
    {
        if (data_)
        {
            assert(0 != nx_ * ny_ * nz_);
            std::free(data_);
        }

        data_ = 0;
        nx_ = 0;
        ny_ = 0;
        nz_ = 0;
    }

    T* data() const noexcept
    {
        return data_;
    }
    T* data() noexcept
    {
        return data_;
    }

    T const& operator()(size_type i, size_type j, size_type k) const noexcept
    {
        return data_[index(i, j, k)];
    }
    T& operator()(size_type i, size_type j, size_type k) noexcept
    {
        return data_[index(i, j, k)];
    }

    T const* operator()(placeholder p, size_type j, size_type k) const noexcept
    {
        return &data_[index(p, j, k)];
    }
    T* operator()(placeholder p, size_type j, size_type k) noexcept
    {
        return &data_[index(p, j, k)];
    }

    T const* operator()(size_type i, placeholder p, size_type k) const noexcept
    {
        return &data_[index(i, p, k)];
    }
    T* operator()(size_type i, placeholder p, size_type k) noexcept
    {
        return &data_[index(i, p, k)];
    }

    T const* operator()(size_type i, size_type j, placeholder p) const noexcept
    {
        return &data_[index(i, j, p)];
    }
    T* operator()(size_type i, size_type j, placeholder p) noexcept
    {
        return &data_[index(i, j, p)];
    }

    constexpr size_type index(
        size_type i, size_type j, size_type k
        ) const noexcept
    {
        return i + nx_ * j + nx_ * ny_ * k;
    }
    constexpr size_type index(
        placeholder, size_type j, size_type k
        ) const noexcept
    {
        return nx_ * j + nx_ * ny_ * k;
    }
    constexpr size_type index(
        size_type i, placeholder, size_type k
        ) const noexcept
    {
        return i + nx_ * ny_ * k;
    }
    constexpr size_type index(
        size_type i, size_type j, placeholder
        ) const noexcept
    {
        return i + nx_ * j;
    }    

    constexpr size_type stride_x() const noexcept
    {
        return 1;
    }
    constexpr size_type stride_y() const noexcept
    {
        return nx_;
    }
    constexpr size_type stride_z() const noexcept
    {
        return nx_ * ny_;
    }

    constexpr size_type nx() const noexcept
    {
        return nx_;
    }
    constexpr size_type ny() const noexcept
    {
        return ny_;
    }
    constexpr size_type nz() const noexcept
    {
        return nz_;
    }
};

///////////////////////////////////////////////////////////////////////////////
// dest = src

inline void copy(
    array3d<float>::size_type j_begin
  , array3d<float>::size_type j_end
  , array3d<float>& dest
  , array3d<float> const& src
    ) noexcept
{
    array3d<float>::size_type const nx = src.nx();
    array3d<float>::size_type const nz = src.nz();

    __assume(0 == (nx % 8)); 

    assert(dest.nx() == src.nx());
    assert(dest.ny() == src.ny());
    assert(dest.nz() == src.nz());

    for (int k = 0; k < nz; ++k)
        for (int j = j_begin; j < j_end; ++j)
        {
            float*       destp = dest(_, j, k);
            float const* srcp  = src (_, j, k);

            __assume_aligned(destp, 64);
            __assume_aligned(srcp, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
                destp[i] = srcp[i];
        }
}

inline void copy(array3d<float>& dest, array3d<float> const& src) noexcept
{
    copy(0, src.ny(), dest, src);
}

///////////////////////////////////////////////////////////////////////////////
// r = A*u - r

inline void residual(
    array3d<float>::size_type j_begin
  , array3d<float>::size_type j_end
  , array3d<float>& r       // Residual
  , array3d<float> const& a // Lower band
  , array3d<float> const& b // Diagonal
  , array3d<float> const& c // Upper band
  , array3d<float> const& u // Solution
    ) noexcept
{
    array3d<float>::size_type const nx = r.nx();
    array3d<float>::size_type const nz = r.nz();

    __assume(0 == (nx % 8)); 

    assert(r.nx()     == a.nx());
    assert(r.ny()     == a.ny());
    assert(r.nz() - 1 == a.nz());

    assert(r.nx()     == b.nx());
    assert(r.ny()     == b.ny());
    assert(r.nz()     == b.nz());

    assert(r.nx()     == c.nx());
    assert(r.ny()     == c.ny());
    assert(r.nz() - 1 == c.nz());

    assert(r.nx()     == u.nx());
    assert(r.ny()     == u.ny());
    assert(r.nz()     == u.nz());

    // First row.
    for (int j = j_begin; j < j_end; ++j)
    {
        float*       r0p = r(_, j, 0);

        float const* b0p = b(_, j, 0);

        float const* c0p = c(_, j, 0);

        float const* u0p = u(_, j, 0);
        float const* u1p = u(_, j, 1);

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
            float*       rp     = r(_, j, k);

            float const* asub1p = a(_, j, k - 1);

            float const* bp     = b(_, j, k);

            float const* cp     = c(_, j, k);

            float const* usub1p = u(_, j, k - 1);
            float const* up     = u(_, j, k);
            float const* uadd1p = u(_, j, k + 1);

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
        float*       rnz1p = r(_, j, nz - 1);

        float const* anz2p = a(_, j, nz - 2);

        float const* bnz1p = b(_, j, nz - 1);

        float const* unz2p = u(_, j, nz - 2);
        float const* unz1p = u(_, j, nz - 1);

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
    array3d<float>& r       // Residual
  , array3d<float> const& a // Lower band
  , array3d<float> const& b // Diagonal
  , array3d<float> const& c // Upper band
  , array3d<float> const& u // Solution
    ) noexcept
{
    residual(0, r.ny(), r, a, b, c, u);
}

///////////////////////////////////////////////////////////////////////////////
// A*x = u; u = x

inline void tridiagonal_solve_native(
    array3d<float>::size_type j_begin
  , array3d<float>::size_type j_end
  , array3d<float>& a    // Lower band
  , array3d<float>& b    // Diagonal
  , array3d<float>& c    // Upper band
  , array3d<float>& u    // Solution
    ) noexcept
{
    array3d<float>::size_type const nx = u.nx();
    array3d<float>::size_type const nz = u.nz();

    __assume(0 == (nx % 8)); 

    assert(u.nx()     == a.nx());
    assert(u.ny()     == a.ny());
    assert(u.nz() - 1 == a.nz());

    assert(u.nx()     == b.nx());
    assert(u.ny()     == b.ny());
    assert(u.nz()     == b.nz());

    assert(u.nx()     == c.nx());
    assert(u.ny()     == c.ny());
    assert(u.nz() - 1 == c.nz());

    // Forward elimination. 
    for (int k = 1; k < nz; ++k)
        for (int j = j_begin; j < j_end; ++j)
        {
            float* asub1p = a(_, j, k - 1);

            float* bp     = b(_, j, k);
            float* bsub1p = b(_, j, k - 1);

            float* csub1p = c(_, j, k - 1);

            float* up     = u(_, j, k);
            float* usub1p = u(_, j, k - 1);

            __assume_aligned(asub1p, 64);

            __assume_aligned(bp, 64);
            __assume_aligned(bsub1p, 64);

            __assume_aligned(csub1p, 64);

            __assume_aligned(up, 64);
            __assume_aligned(usub1p, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                // float const m = a[k - 1] / b[k - 1];
                float const m = asub1p[i] / bsub1p[i];
                // b[k] -= m * c[k - 1];
                bp[i] -= m * csub1p[i];
                // u[k] -= m * u[k - 1];
                up[i] -= m * usub1p[i];
            }
        }

    for (int j = j_begin; j < j_end; ++j)
    {
        float* bendp = b(_, j, nz - 1);

        float* uendp = u(_, j, nz - 1);

        __assume_aligned(bendp, 64);

        __assume_aligned(uendp, 64);

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // u[nz - 1] = u[nz - 1] / b[nz - 1];
            uendp[i] = uendp[i] / bendp[i];
        }
    }
 
    // Back substitution. 
    for (int k = nz - 2; k >= 0; --k)
        for (int j = j_begin; j < j_end; ++j)
        {
            float* bp     = b(_, j, k);

            float* cp     = c(_, j, k);

            float* up     = u(_, j, k);
            float* uadd1p = u(_, j, k + 1);

            __assume_aligned(bp, 64);

            __assume_aligned(cp, 64);

            __assume_aligned(up, 64);
            __assume_aligned(uadd1p, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                // u[k] = (u[k] - c[k] * u[k + 1]) / b[k];
                up[i] = (up[i] - cp[i] * uadd1p[i]) / bp[i];
            }
        }
}

///////////////////////////////////////////////////////////////////////////////

struct heat_equation_btcs
{
    bool verify;
    bool header;

    float D;

    std::uint64_t nx;
    std::uint64_t ny;
    std::uint64_t nz;
    std::uint64_t tw;

    std::uint64_t ns;
    float dt;

  private:
    float dz;

    static float constexpr N = 1.0;

    float A_coef;

    array3d<float> a;
    array3d<float> b;
    array3d<float> c;

    array3d<float> u;

    array3d<float> r;

  public:
    heat_equation_btcs() noexcept
    {
        verify = get_env_variable<bool>("verify", false);
        header = get_env_variable<bool>("header", false);

        D  = get_env_variable<float>("D", 0.1);

        nx = get_env_variable<std::uint64_t>("nx", 32);
        ny = get_env_variable<std::uint64_t>("ny", 2048);
        nz = get_env_variable<std::uint64_t>("nz", 32);
        tw = get_env_variable<std::uint64_t>("tw", 16);

        ns = get_env_variable<std::uint64_t>("ns", 50);
        dt = get_env_variable<float>("dt", 1.0e-7);

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

        // Allocate storage for the matrix.
        a.resize(nx, ny, nz - 1);
        b.resize(nx, ny, nz);
        c.resize(nx, ny, nz - 1);

        // Allocate storage for the problem state and initialize it.
        u.resize(nx, ny, nz);

        __assume(0 == (nx % 8));

        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            {
                float* up = u(_, j, k);

                __assume_aligned(up, 64);

                #pragma simd
                for (int i = 0; i < nx; ++i)
                    up[i] = std::sin(N * M_PI * (dz * k));
            }

        // Allocate storage for the residual.
        r.resize(nx, ny, nz);
    }

    void build_matrix(std::ptrdiff_t j_begin, std::ptrdiff_t j_end) noexcept
    {
        float const ac_term = -A_coef;
        float const b_term  = 1.0 + 2.0 * A_coef;

        __assume(0 == (nx % 8)); 
 
        for (int k = 0; k < nz; ++k)
            for (int j = j_begin; j < j_end; ++j)
            {
                float* bp = b(_, j, k);

                __assume_aligned(bp, 64);

                #pragma simd
                for (int i = 0; i < nx; ++i)
                    bp[i] = b_term;
            }

        for (int k = 0; k < nz - 1; ++k)
            for (int j = j_begin; j < j_end; ++j)
            {
                float* ap = a(_, j, k);
                float* cp = c(_, j, k);

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
            float* bbeginp = b(_, j, 0);
            float* cbeginp = c(_, j, 0);

            float* aendp   = a(_, j, nz - 2);
            float* bendp   = b(_, j, nz - 1);

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

    float l2_norm(std::uint64_t step) noexcept
    {
        float l2 = 0.0;

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                float sum = 0.0;

                float const* up = u(i, j, _);

                array3d<float>::size_type const stride = u.stride_z();

                __assume_aligned(up, 64);

                // NOTE: Strided access.
                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    array3d<float>::size_type const ks = k * stride;

                    float const exact = std::exp( -D * (N * N)
                                                 * (M_PI * M_PI) * (dt * step))
                                       * std::sin(N * M_PI * (dz * k)); 

                    float const abs_term = std::fabs(up[ks] - exact);
                    sum = sum + abs_term * abs_term;
                }

                float const l2_here = std::sqrt(sum);

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

    float max_residual() noexcept
    {
        residual(r, a, b, c, u);

        float mr = 0.0; 

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                float min = 1.0e30;
                float max = -1.0e30;

                float const* rp = r(i, j, _);

                array3d<float>::size_type const stride = r.stride_z();

                __assume_aligned(rp, 64);

                // NOTE: Strided access.
                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    array3d<float>::size_type const ks = k * stride;

                    min = std::min(min, rp[ks]);
                    max = std::max(max, rp[ks]);
                }

                float const mr_here = std::max(std::fabs(min), std::fabs(max));

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

        for (int s = 0; s < ns; ++s)
        {
            if (verify)
                copy(r, u);

            for (int j = 0; j < ny; j += tw)
            {
                std::ptrdiff_t j_begin = j;
                std::ptrdiff_t j_end   = j + tw;

                build_matrix(j_begin, j_end);

                tridiagonal_solve_native(j_begin, j_end, a, b, c, u);
            }

            if (verify)
            {
                for (int j = 0; j < ny; j += tw)
                {
                    std::ptrdiff_t j_begin = j;
                    std::ptrdiff_t j_end   = j + tw;

                    build_matrix(j_begin, j_end);
                }

                float const resid = max_residual();

                float const l2 = l2_norm(s + 1);

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

        float const walltime = t.elapsed();

        float const l2 = l2_norm(ns);  

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
                "Walltime [s],"
                "L2 Norm"
                ;

        std::cout
            << "Streaming Single Precision,"
            << BUILD_TYPE << ","
            << D << ","
            << nx << ","
            << ny << ","
            << nz << ","
            << tw << ","
            << ns << ","
            << dt << ","
            << std::setprecision(7) << walltime << ","
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

