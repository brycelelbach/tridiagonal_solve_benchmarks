#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include <cmath>
#include <cassert>
#include <cstddef>
#include <cstdint>

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

struct placeholder {};

constexpr placeholder _{};

template <typename T, std::size_t Alignment = 64>
struct array3d
{
    typedef std::ptrdiff_t size_type;
    typedef T value_type;

  private:
    T* data_;
    size_type nx_, ny_, nz_;

  public:
    array3d() : data_(), nx_(), ny_(), nz_() {}

    array3d(size_type nx, size_type ny, size_type nz)
    {
        resize(nx, ny, nz);
    }

    ~array3d()
    {
        clear();
    }

    void resize(size_type nx, size_type ny, size_type nz)
    {
        clear();

        void* p = 0; 
        int const result
            = posix_memalign(&p, Alignment, nx * ny * nz * sizeof(T));
        assert(0 == result);
        data_ = reinterpret_cast<double*>(p);

        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
    }

    void clear()
    {
        if (data_)
        {
            assert(0 != nx_ * ny_ * nz_);
            free(data_);
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

    size_type index(size_type i, size_type j, size_type k) const noexcept
    {
        return i + nx_ * j + nx_ * ny_ * k;
    }    
    size_type index(size_type j, size_type k) const noexcept
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

void tridiagonal_solve_native(
    array3d<double>& a, // Lower band
    array3d<double>& b, // Diagonal
    array3d<double>& c, // Upper band
    array3d<double>& u  // Solution
    )
{
    auto const nx = u.nx();
    auto const ny = u.ny();
    auto const nz = u.nz();

    // Forward elimination. 
    for (int k = 1; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        {
            double* up     = u(_, j, k);
            double* usub1p = u(_, j, k - 1);

            double* asub1p = a(_, j, k - 1);

            double* bp     = b(_, j, k);
            double* bsub1p = b(_, j, k - 1);

            double* csub1p = c(_, j, k - 1);

            __assume_aligned(up, 64);
            __assume_aligned(usub1p, 64);

            __assume_aligned(asub1p, 64);

            __assume_aligned(bp, 64);
            __assume_aligned(bsub1p, 64);

            __assume_aligned(csub1p, 64);

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

    for (int j = 0; j < ny; ++j)
    {
        double* uendp = u(_, j, nz - 1);

        double* bendp = b(_, j, nz - 1);

        __assume_aligned(uendp, 64);

        __assume_aligned(bendp, 64);

        #pragma simd
        for (int i = 0; i < nx; ++i)
        {
            // u[nz - 1] = u[nz - 1] / b[nz - 1];
            uendp[i] = uendp[i] / bendp[i];
        }
    }
 
    // Back substitution. 
    for (int k = nz - 2; k >= 0; --k)
        for (int j = 0; j < ny; ++j)
        {
            double* up     = u(_, j, k);
            double* uadd1p = u(_, j, k + 1);

            double* bp     = b(_, j, k);

            double* cp     = c(_, j, k);

            __assume_aligned(up, 64);
            __assume_aligned(uadd1p, 64);

            __assume_aligned(bp, 64);

            __assume_aligned(cp, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                // u[k] = (u[k] - c[k] * u[k + 1]) / b[k];
                up[i] = (up[i] - cp[i] * uadd1p[i]) / bp[i];
            }
        }
}

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
//
struct heat_equation_btcs
{
    double D;
    double N;

    std::uint64_t nx;
    std::uint64_t ny;
    std::uint64_t nz;
    std::uint64_t ns;
    double nt;

  private:
    double dz;
    double dt;

    double r;

    array3d<double> a;
    array3d<double> b;
    array3d<double> c;

    array3d<double> u;    

    array3d<double> error;

  public:
    heat_equation_btcs()
    {
        D = 0.1;
        N = 1.0;
        nx = 128;
        ny = 128;
        nz = 32;
        ns = 200;
        nt = 0.002;
    }

    void initialize()
    { 
        // Compute time-step size and grid spacing.
        dz = 1.0 / (nz - 1);
        dt = nt / (ns - 1);

        // Compute matrix constant.
        r = D * dt / (dz * dz);

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

        // Allocate storage for storing error calculations.
        error.resize(nx, ny, nz);
    }

    void build_matrix()
    {
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            {
                double* bp = b(_, j, k);
                for (int i = 0; i < nx; ++i)
                    bp[i] = 1.0 + 2.0 * r;
            }

        for (int k = 0; k < nz - 1; ++k)
            for (int j = 0; j < ny; ++j)
            {
                double* ap = a(_, j, k);
                double* cp = c(_, j, k);
                for (int i = 0; i < nx; ++i)
                {
                    ap[i] = -r;
                    cp[i] = -r;
                }
            }

        // Boundary conditions.
        for (int j = 0; j < ny; ++j)
        {
            double* bbeginp = b(_, j, 0);
            double* cbeginp = c(_, j, 0);

            double* aendp   = a(_, j, nz - 2);
            double* bendp   = b(_, j, nz - 1);

            for (int i = 0; i < nx; ++i)
            {
                bbeginp[i] = 1.0; cbeginp[i] = 0.0;
                aendp[i]   = 0.0; bendp[i]   = 1.0;
            }
        }
    }

    void solve()
    {
        initialize();

        high_resolution_timer t;

        for (int s = 1; s < ns; ++s)
        {
            build_matrix();

            tridiagonal_solve_native(a, b, c, u);
        }

        double const walltime = t.elapsed();

        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            {
                double* up     = u(_, j, k);
                double* errorp = error(_, j, k);
                for (int i = 0; i < nx; ++i)
                {
                    double exact = std::exp( -D * (N * N)
                                           * (M_PI * M_PI) * (dt * (ns - 1)))
                                 * std::sin(N * M_PI * (dz * k)); 

                    errorp[i] = (up[i] - exact); 
                }
            }

        double sum = 0.0;

        for (int k = 0; k < nz; ++k)
        {
            double const abs_term = std::fabs(error(0, 0, k));
            sum = sum + abs_term * abs_term;
        }

        double l2_norm = std::sqrt(sum);

        std::cout << std::setprecision(16)
                  << "WALLTIME == " << walltime << " [s]\n"
                  << "L2 NORM  == " << l2_norm << "\n";
    }
};

int main()
{
    std::cout << "SOLVER: NATIVE\n";

    heat_equation_btcs s;

    s.solve();
}

