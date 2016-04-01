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

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <type_traits>

#include <cmath>
#include <cstring>
#include <cassert>
#include <climits>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// FIXME: Some of these can be noexcept/constexpr

struct high_resolution_timer
{
    high_resolution_timer()
      : start_time_(take_time_stamp())
    {}

    void restart()
    {
        start_time_ = take_time_stamp();
    }

    double elapsed() const // Return elapsed time in seconds.
    {
        return double(take_time_stamp() - start_time_) * 1e-9;
    }

    std::uint64_t elapsed_nanoseconds() const
    {
        return take_time_stamp() - start_time_;
    }

  protected:
    static std::uint64_t take_time_stamp()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::steady_clock::now().time_since_epoch()).count();
    }

  private:
    std::uint64_t start_time_;
};

template <typename T>
T get_env_variable(std::string const& var, T default_val); 

template <>
bool get_env_variable(std::string const& var, bool default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    std::uint64_t const r = std::strtoul(env_str.c_str(), &env_str_p_end, 10);

    if ((&env_str.back() != env_str_p_end - 1) || ULONG_MAX == r)
    {
        std::cout << "ERROR: invalid value '" << env_str << "' "
                     "for boolean environment variable '" << var << "'"
                     "\n";
        std::exit(1);
    }

    return bool(r);
}

template <>
std::uint64_t get_env_variable(std::string const& var, std::uint64_t default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    std::uint64_t const r = std::strtoul(env_str.c_str(), &env_str_p_end, 10);

    if ((&env_str.back() != env_str_p_end - 1) || ULONG_MAX == r)
    {
        std::cout << "ERROR: invalid value '" << env_str << "' "
                     "for integer environment variable '" << var << "'"
                     "\n";
        std::exit(1);
    }

    return r;
}

template <>
double get_env_variable(std::string const& var, double default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    double const r = std::strtod(env_str.c_str(), &env_str_p_end);

    if ((&env_str.back() != env_str_p_end - 1) || ULONG_MAX == r)
    {
        std::cout << "ERROR: invalid value '" << env_str << "' "
                     "for floating point environment variable '" << var << "'"
                     "\n";
        std::exit(1);
    }

    return r;
}

template <
    typename T
  , typename = typename std::enable_if<std::is_floating_point<T>::value>::type
    >
constexpr bool fp_equals(
    T x, T y, T epsilon = std::numeric_limits<T>::epsilon()
    ) noexcept
{
    return ( ((x + epsilon >= y) && (x - epsilon <= y))
           ? true
           : false);
}

struct placeholder {};

constexpr placeholder _{};

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

        data_ = reinterpret_cast<double*>(p);

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

    T const& operator()(size_type i, size_type j, size_type k) const noexcept
    {
        return data_[index(i, j, k)];
    }
    T& operator()(size_type i, size_type j, size_type k) noexcept
    {
        return data_[index(i, j, k)];
    }

    T const* operator()(placeholder, size_type j, size_type k) const noexcept
    {
        return &data_[index(j, k)];
    }
    T* operator()(placeholder, size_type j, size_type k) noexcept
    {
        return &data_[index(j, k)];
    }

    T* data() const noexcept
    {
        return data_;
    }
    T* data() noexcept
    {
        return data_;
    }

    constexpr size_type index(
        size_type i, size_type j, size_type k
        ) const noexcept
    {
        return i + nx_ * j + nx_ * ny_ * k;
    }    
    constexpr size_type index(
        size_type j, size_type k
        ) const noexcept
    {
        return nx_ * j + nx_ * ny_ * k;
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

            __assume(0 == (nx % 8)); 

            #pragma simd
            for (int i = 0; i < nx; ++i)
                destp[i] = srcp[i];
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

        __assume(0 == (nx % 8)); 

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

            __assume(0 == (nx % 8)); 

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

        __assume(0 == (nx % 8)); 

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
// A*x = u; u = x

inline void tridiagonal_solve_native(
    array3d<double>::size_type j_begin
  , array3d<double>::size_type j_end
  , array3d<double>& a    // Lower band
  , array3d<double>& b    // Diagonal
  , array3d<double>& c    // Upper band
  , array3d<double>& u    // Solution
    ) noexcept
{
    array3d<double>::size_type const nx = u.nx();
    array3d<double>::size_type const nz = u.nz();

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

            __assume(0 == (nx % 8)); 

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                // double const m = a[k - 1] / b[k - 1];
                double const m = asub1p[i] / bsub1p[i];
                // b[k] -= m * c[k - 1];
                bp[i] -= m * csub1p[i];
                // u[k] -= m * u[k - 1];
                up[i] -= m * usub1p[i];
            }
        }

    for (int j = j_begin; j < j_end; ++j)
    {
        double* bendp = b(_, j, nz - 1);

        double* uendp = u(_, j, nz - 1);

        __assume_aligned(bendp, 64);

        __assume_aligned(uendp, 64);

        __assume(0 == (nx % 8)); 

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
            double* bp     = b(_, j, k);

            double* cp     = c(_, j, k);

            double* up     = u(_, j, k);
            double* uadd1p = u(_, j, k + 1);

            __assume_aligned(bp, 64);

            __assume_aligned(cp, 64);

            __assume_aligned(up, 64);
            __assume_aligned(uadd1p, 64);

            __assume(0 == (nx % 8)); 

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

    double D;
    double N;

    std::uint64_t nx;
    std::uint64_t ny;
    std::uint64_t nz;
    std::uint64_t ns;
    std::uint64_t tw;
    double nt;

  private:
    double dz;
    double dt;

    double coef;

    array3d<double> a;
    array3d<double> b;
    array3d<double> c;

    array3d<double> u;

    array3d<double> r;

    array3d<double> error;

  public:
    heat_equation_btcs() noexcept
    {
        verify = get_env_variable<bool>("verify", false);

        D  = get_env_variable<double>("D", 0.1);
        N  = get_env_variable<double>("N", 1.0);
        nx = get_env_variable<std::uint64_t>("nx", 128);
        ny = get_env_variable<std::uint64_t>("ny", 128);
        nz = get_env_variable<std::uint64_t>("nz", 32);
        ns = get_env_variable<std::uint64_t>("ns", 200);
        tw = get_env_variable<std::uint64_t>("tw", 8);
        nt = get_env_variable<double>("nt", 0.002);

        assert(0 == (nx % 8)); // Ensure 64-byte alignment.
        assert(8 <= nx); 

        assert(0 == (ny % tw));
    }

    void initialize() noexcept 
    { 
        // Compute time-step size and grid spacing.
        dz = 1.0 / (nz - 1);
        dt = nt / ns;

        // Compute matrix constant.
        coef = D * dt / (dz * dz);

        // Allocate storage for the matrix.
        a.resize(nx, ny, nz - 1);
        b.resize(nx, ny, nz);
        c.resize(nx, ny, nz - 1);

        // Allocate storage for the problem state and initialize it.
        u.resize(nx, ny, nz);
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            {
                double* up = u(_, j, k);
                for (int i = 0; i < nx; ++i)
                    up[i] = std::sin(N * M_PI * (dz * k));
            }

        // Allocate storage for the residual.
        r.resize(nx, ny, nz);

        // Allocate storage for error calculations.
        error.resize(nx, ny, nz);
    }

    void build_matrix(std::ptrdiff_t j_begin, std::ptrdiff_t j_end) noexcept
    {
        double const ac_term = -coef;
        double const b_term  = 1.0 + 2.0 * coef;
 
        for (int k = 0; k < nz; ++k)
            for (int j = j_begin; j < j_end; ++j)
            {
                double* bp = b(_, j, k);

                __assume_aligned(bp, 64);

                __assume(0 == (nx % 8)); 

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

                __assume(0 == (nx % 8)); 

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

            __assume(0 == (nx % 8)); 

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                bbeginp[i] = 1.0; cbeginp[i] = 0.0;
                aendp[i]   = 0.0; bendp[i]   = 1.0;
            }
        }
    }

    double l2_norm(std::uint64_t step)
    {
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            {
                double* up     = u(_, j, k);
                double* errorp = error(_, j, k);
                for (int i = 0; i < nx; ++i)
                {
                    double exact = std::exp( -D * (N * N)
                                           * (M_PI * M_PI) * (dt * step))
                                 * std::sin(N * M_PI * (dz * k)); 

                    errorp[i] = (up[i] - exact); 
                }
            }

        double e = 0.0;

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                double sum = 0.0;

                for (int k = 0; k < nz; ++k)
                {
                    double const abs_term = std::fabs(error(i, j, k));
                    sum = sum + abs_term * abs_term;
                }

                double const e_here = std::sqrt(sum);

                if ((0 == i) && (0 == j))
                    // First iteration, so we have nothing to compare against.
                    e = e_here;
                else
                    // All the columns are the same, so the L2 norm for each
                    // column should be the same.
                    assert(fp_equals(e, e_here));
            }

        return e;
    }

    double max_residual()
    {
        residual(r, a, b, c, u);

        double mr = 0.0; 

        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                double min = 1.0e300;
                double max = -1.0e300;

                for (int k = 0; k < nz; ++k)
                {
                    min = std::min(min, r(i, j, k));
                    max = std::max(max, r(i, j, k));
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

                tridiagonal_solve_native(j_begin, j_end, a, b, c, u);
            }

            if (verify)
            {
                double const resid = max_residual();

                double const l2 = l2_norm(s + 1);  

                std::printf(
                    "STEP %04u : "
                    "TIME %10.7g = %10.7g + %10.7g : "
                    "L2 NORM %22.16g : "
                    "MAX RESIDUAL %22.16g\n"
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

        std::cout << "WALLTIME : " << walltime << " [s]\n";
    }
};

int main()
{
    std::cout << "SOLVER : NATIVE\n";

    heat_equation_btcs s;

    s.solve();
}

