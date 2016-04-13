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
        return nz_ * ny_ * i + nz_ * j + k;
    }
    constexpr size_type index(
        placeholder, size_type j, size_type k
        ) const noexcept
    {
        return nz_ * j + k;
    }
    constexpr size_type index(
        size_type i, placeholder, size_type k
        ) const noexcept
    {
        return nz_ * ny_ * i + k;
    }
    constexpr size_type index(
        size_type i, size_type j, placeholder
        ) const noexcept
    {
        return nz_ * ny_ * i + nz_ * j;
    }

    constexpr size_type stride_x() const noexcept
    {
        return nz_ * ny_;
    }
    constexpr size_type stride_y() const noexcept
    {
        return nz_;
    }
    constexpr size_type stride_z() const noexcept
    {
        return 1;
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
    array3d<double>::size_type j_begin
  , array3d<double>::size_type j_end
  , array3d<double>& dest
  , array3d<double> const& src
    ) noexcept
{
    array3d<double>::size_type const nx = src.nx();
    array3d<double>::size_type const nz = src.nz();

    __assume(0 == (nz % 8)); 

    assert(dest.nx() == src.nx());
    assert(dest.ny() == src.ny());
    assert(dest.nz() == src.nz());

    for (int i = 0; i < nx; ++i)
        for (int j = j_begin; j < j_end; ++j)
        {
            double*       destp = dest(i, j, _);
            double const* srcp  = src (i, j, _);

            __assume_aligned(destp, 64);
            __assume_aligned(srcp, 64);

            #pragma simd
            for (int k = 0; k < nz; ++k)
                destp[k] = srcp[k];
        }
}

inline void copy(array3d<double>& dest, array3d<double> const& src) noexcept
{
    copy(0, src.ny(), dest, src);
}

///////////////////////////////////////////////////////////////////////////////
// r = A*u - r

inline void residual(
    array3d<double>::size_type j_begin
  , array3d<double>::size_type j_end
  , array3d<double>& r       // Residual
  , array3d<double> const& a // Lower band
  , array3d<double> const& b // Diagonal
  , array3d<double> const& c // Upper band
  , array3d<double> const& u // Solution
    ) noexcept
{
    array3d<double>::size_type const nx = r.nx();
    array3d<double>::size_type const nz = r.nz();

    __assume(0 == (nz % 8)); 

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
    for (int i = 0; i < nx; ++i)
    {
        double*       r0p = r(i, _, 0);

        double const* b0p = b(i, _, 0);

        double const* c0p = c(i, _, 0);

        double const* u0p = u(i, _, 0);
        double const* u1p = u(i, _, 1);

        array3d<double>::size_type const ac_stride  = a.stride_y();
        array3d<double>::size_type const bur_stride = b.stride_y();

        __assume_aligned(r0p, 64);

        __assume_aligned(b0p, 64);

        __assume_aligned(c0p, 64);

        __assume_aligned(u0p, 64);
        __assume_aligned(u1p, 64);

        // NOTE: Strided access.
        #pragma simd
        for (int j = j_begin; j < j_end; ++j)
        {
            array3d<double>::size_type const jacs  = j * ac_stride;
            array3d<double>::size_type const jburs = j * bur_stride;

            // NOTE: The comment is k-indexed. The code is j-indexed.
            // r[0] = (b[0] * u[0] + c[0] * u[1]) - r[0];
            r0p[jburs] = (b0p[jburs] * u0p[jburs] + c0p[jacs] * u1p[jburs])
                       - r0p[jburs];
        }
    }

    // Interior rows.
    for (int i = 0; i < nx; ++i)
        for (int j = j_begin; j < j_end; ++j)
        {
            double*       rp = r(i, j, _);

            double const* ap = a(i, j, _);

            double const* bp = b(i, j, _);

            double const* cp = c(i, j, _);

            double const* up = u(i, j, _);

            __assume_aligned(rp, 64);

            __assume_aligned(ap, 64);

            __assume_aligned(bp, 64);

            __assume_aligned(cp, 64);

            __assume_aligned(up, 64);

            #pragma simd
            for (int k = 1; k < nz - 1; ++k)
            {
                // r[k] = ( a[k - 1] * u[k - 1]
                //        + b[k] * u[k]
                //        + c[k] * u[k + 1])
                //      - r[k];
                rp[k] = ( ap[k - 1] * up[k - 1]
                        + bp[k] * up[k]
                        + cp[k] * up[k + 1])
                      - rp[k];
            }
        }

    // Last row.
    for (int i = 0; i < nx; ++i)
    {
        double*       rnz1p = r(i, _, nz - 1);

        double const* anz2p = a(i, _, nz - 2);

        double const* bnz1p = b(i, _, nz - 1);

        double const* unz2p = u(i, _, nz - 2);
        double const* unz1p = u(i, _, nz - 1);

        array3d<double>::size_type const ac_stride  = a.stride_y();
        array3d<double>::size_type const bur_stride = b.stride_y();

        __assume_aligned(rnz1p, 64);

        __assume_aligned(anz2p, 64);

        __assume_aligned(bnz1p, 64);

        __assume_aligned(unz2p, 64);
        __assume_aligned(unz1p, 64);

        // NOTE: Strided access.
        #pragma simd
        for (int j = j_begin; j < j_end; ++j)
        {
            array3d<double>::size_type const jacs  = j * ac_stride;
            array3d<double>::size_type const jburs = j * bur_stride;

            // NOTE: The comment is k-indexed. The code is j-indexed.
            // r[nz - 1] = (a[nz - 2] * u[nz - 2] + b[nz - 1] * u[nz - 1])
            //           - r[nz - 1];
            rnz1p[jburs] = ( anz2p[jacs] * unz2p[jburs]
                           + bnz1p[jburs] * unz1p[jburs])
                         - rnz1p[jburs];
        }
    }
}

inline void residual(
    array3d<double>& r       // Residual
  , array3d<double> const& a // Lower band
  , array3d<double> const& b // Diagonal
  , array3d<double> const& c // Upper band
  , array3d<double> const& u // Solution
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

    array3d<double> a;
    array3d<double> b;
    array3d<double> c;

    array3d<double> u;

    array3d<double> r;

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
        a.resize(nx, ny, nz - 1);
        b.resize(nx, ny, nz);
        c.resize(nx, ny, nz - 1);

        // Allocate storage for the problem state and initialize it.
        u.resize(nx, ny, nz);

        __assume(0 == (nz % 8));

        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                double* up = u(i, j, _);

                __assume_aligned(up, 64);

                #pragma simd
                for (int k = 0; k < nz; ++k)
                    up[k] = std::sin(N * M_PI * (dz * k));
            }

        // Allocate storage for the residual.
        r.resize(nx, ny, nz);
    }

    void build_matrix(std::ptrdiff_t j_begin, std::ptrdiff_t j_end) noexcept
    {
        double const ac_term = -A_coef;
        double const b_term  = 1.0 + 2.0 * A_coef;

        __assume(0 == (nz % 8)); 

        for (int i = 0; i < nx; ++i)
            for (int j = j_begin; j < j_end; ++j)
            {
                double* bp = b(i, j, _);

                __assume_aligned(bp, 64);

                #pragma simd
                for (int k = 0; k < nz; ++k)
                    bp[k] = b_term;

                double* ap = a(i, j, _);
                double* cp = c(i, j, _);

                __assume_aligned(ap, 64);
                __assume_aligned(cp, 64);

                #pragma simd
                for (int k = 0; k < nz - 1; ++k)
                {
                    ap[k] = ac_term;
                    cp[k] = ac_term;
                }
            }

        // Boundary conditions.
        for (int i = 0; i < nx; ++i)
        {
            double* bbeginp = b(i, _, 0);
            double* cbeginp = c(i, _, 0);

            double* aendp   = a(i, _, nz - 2);
            double* bendp   = b(i, _, nz - 1);

            array3d<double>::size_type const ac_stride = a.stride_y();
            array3d<double>::size_type const b_stride  = b.stride_y();

            __assume_aligned(bbeginp, 64);
            __assume_aligned(cbeginp, 64);

            __assume_aligned(aendp, 64);
            __assume_aligned(bendp, 64);

            // NOTE: Strided access.
            #pragma simd
            for (int j = j_begin; j < j_end; ++j)
            {
                array3d<double>::size_type const jacs = j * ac_stride;
                array3d<double>::size_type const jbs  = j * b_stride;

                bbeginp[jbs] = 1.0; cbeginp[jacs] = 0.0;
                aendp[jacs]  = 0.0; bendp[jbs]    = 1.0;
            }
        }
    }

    double l2_norm(std::uint64_t step) noexcept
    {
        double l2 = 0.0;

        __assume(0 == (nz % 8)); 

        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                double sum = 0.0;

                double const* up = u(i, j, _);

                __assume_aligned(up, 64);

                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    double const exact = std::exp( -D * (N * N)
                                                 * (M_PI * M_PI) * (dt * step))
                                       * std::sin(N * M_PI * (dz * k)); 

                    double const abs_term = std::fabs(up[k] - exact);
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

        __assume(0 == (nz % 8)); 

        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                double min = 1.0e300;
                double max = -1.0e300;

                double const* rp = r(i, j, _);

                __assume_aligned(rp, 64);

                #pragma simd
                for (int k = 0; k < nz; ++k)
                {
                    min = std::min(min, rp[k]);
                    max = std::max(max, rp[k]);
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

        for (int s = 0; s < ns; ++s)
        {
            if (verify)
                copy(r, u);

            for (int j = 0; j < ny; j += tw)
            {
                std::ptrdiff_t j_begin = j;
                std::ptrdiff_t j_end   = j + tw;

                build_matrix(j_begin, j_end);
            }

            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                {
                    int mkl_n    = nz;
                    int mkl_nrhs = 1;
                    int mkl_ldb  = nz;
                    int mkl_info = 0;

                    dgtsv_(
                        &mkl_n,       // Matrix order.
                        &mkl_nrhs,    // # of right hand sides.
                        a(i, j, _),   // Subdiagonal part.
                        b(i, j, _),   // Diagonal part.
                        c(i, j, _),   // Superdiagonal part.
                        u(i, j, _),   // Column to solve.
                        &mkl_ldb,     // Leading dimension of RHS.
                        &mkl_info
                        );

                    assert(mkl_info == 0);
                }

            if (verify)
            {
                for (int j = 0; j < ny; j += tw)
                {
                    std::ptrdiff_t j_begin = j;
                    std::ptrdiff_t j_end   = j + tw;

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
                "Walltime [s],"
                "L2 Norm"
                ;

        std::cout
            << "MKL Bulk Build Contigous Z,"
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
