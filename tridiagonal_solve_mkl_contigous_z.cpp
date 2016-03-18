#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

#include <cmath>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <mkl.h>

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
struct array3d
{
    typedef std::ptrdiff_t size_type;
    typedef T value_type;

  private:
    std::vector<T> data_;
    size_type nx_, ny_, nz_;

  public:
    array3d() : data_(), nx_(), ny_(), nz_() {}

    array3d(size_type nx, size_type ny, size_type nz)
      : data_(nx * ny * nz)
      , nx_(nx), ny_(ny), nz_(nz)
    {}

    void resize(size_type nx, size_type ny, size_type nz)
    {
        data_.resize(nx * ny * nz);
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;
    }

    T const& operator()(size_type i, size_type j, size_type k) const noexcept
    {
        return data_[index(i, j, k)];
    }
    T& operator()(size_type i, size_type j, size_type k) noexcept
    {
        return data_[index(i, j, k)];
    }

    T const* operator()(size_type i, size_type j) const noexcept
    {
        return &data_[index(i, j)];
    }
    T* operator()(size_type i, size_type j) noexcept
    {
        return &data_[index(i, j)];
    }

    size_type index(size_type i, size_type j, size_type k) const noexcept
    {
        return nz_ * ny_ * i + nz_ * j + k;
    }    
    size_type index(size_type i, size_type j) const noexcept
    {
        return nz_ * ny_ * i + nz_ * j;
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
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                double* up = u(i, j);
                for (int k = 0; k < nz; ++k)
                    up[k] = std::sin(N * M_PI * (dz * k));
            }

        // Allocate storage for storing error calculations.
        error.resize(nx, ny, nz);
    }

    void build_matrix()
    {
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                double* bp = b(i, j);
                for (int k = 0; k < nz; ++k)
                    bp[k] = 1.0 + 2.0 * r;

                double* ap = a(i, j);
                double* cp = c(i, j);
                for (int k = 0; k < nz - 1; ++k)
                {
                    ap[k] = -r;
                    cp[k] = -r;
                }
            }

        // Boundary conditions.
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                b(i, j, 0     ) = 1.0; c(i, j, 0     ) = 0.0;
                a(i, j, nz - 2) = 0.0; b(i, j, nz - 1) = 1.0;
            }
    }

    void solve()
    {
        initialize();

        high_resolution_timer t;

        for (int s = 1; s < ns; ++s)
        {
            build_matrix();

            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                {
                    int mkl_n    = nz;
                    int mkl_nrhs = 1;
                    int mkl_ldb  = nz;
                    int mkl_info = 0;

                    dgtsv_(
                        &mkl_n,       // matrix order
                        &mkl_nrhs,    // # of right hand sides 
                        a(i, j),      // subdiagonal part
                        b(i, j),      // diagonal part
                        c(i, j),      // superdiagonal part
                        u(i, j),      // column to solve 
                        &mkl_ldb,     // leading dimension of RHS
                        &mkl_info
                        );

                    assert(mkl_info == 0);
                }
        }

        double const walltime = t.elapsed();

        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
            {
                double* up     = u(i, j);
                double* errorp = error(i, j);
                for (int k = 0; k < nz; ++k)
                {
                    double exact = std::exp( -D * (N * N)
                                           * (M_PI * M_PI) * (dt * (ns - 1)))
                                 * std::sin(N * M_PI * (dz * k)); 

                    errorp[k] = (up[k] - exact); 
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
    std::cout << "SOLVER: LAPACK\n";

    heat_equation_btcs s;

    s.solve();
}
