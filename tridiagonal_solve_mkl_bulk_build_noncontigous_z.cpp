#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>

#include <cmath>
#include <cfenv>
#include <cstring>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

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
T get_env_variable(std::string const& var, T default_val); 

template <>
std::size_t get_env_variable(std::string const& var, std::size_t default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    std::size_t const r = std::strtoul(env_str.c_str(), &env_str_p_end, 10);

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

        int const r = posix_memalign(&p, Alignment, nx * ny * nz * sizeof(T));
        assert(0 == r);

        std::memset(p, 0, nx * ny * nz * sizeof(T));

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

    T* data() const noexcept
    {
        return data_;
    }
    T* data() noexcept
    {
        return data_;
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

    std::size_t nx;
    std::size_t ny;
    std::size_t nz;
    std::size_t ns;
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

    std::vector<double> a_buf;
    std::vector<double> b_buf;
    std::vector<double> c_buf;

    std::vector<double> u_buf;

  public:
    heat_equation_btcs()
    {
        D  = get_env_variable<double>("D", 0.1);
        N  = get_env_variable<double>("N", 1.0);
        nx = get_env_variable<std::size_t>("nx", 128);
        ny = get_env_variable<std::size_t>("ny", 128);
        nz = get_env_variable<std::size_t>("nz", 32);
        ns = get_env_variable<std::size_t>("ns", 200);
        nt = get_env_variable<double>("nt", 0.002);
    }

    void initialize()
    { 
        // Compute time-step size and grid spacing.
        dz = 1.0 / (nz - 1);
        dt = nt / ns;

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

        // Allocate storage for the matrix gather buffers.
        a_buf.resize(nz - 1);
        b_buf.resize(nz);
        c_buf.resize(nz - 1);

        // Allocate storage for the problem state gather buffers.
        u_buf.resize(nz);
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

        for (int s = 0; s < ns; ++s)
        {
            build_matrix();

            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    for (int k = 0; k < nz; ++k)
                    {
                        b_buf[k] = b(i, j, k);
                        u_buf[k] = u(i, j, k);
                    }

                    for (int k = 0; k < nz - 1; ++k)
                    {
                        a_buf[k] = a(i, j, k);
                        c_buf[k] = c(i, j, k);
                    }

                    int mkl_n    = nz;
                    int mkl_nrhs = 1;
                    int mkl_ldb  = nz;
                    int mkl_info = 0;

                    dgtsv_(
                        &mkl_n,       // matrix order
                        &mkl_nrhs,    // # of right hand sides 
                        a_buf.data(), // subdiagonal part
                        b_buf.data(), // diagonal part
                        c_buf.data(), // superdiagonal part
                        u_buf.data(), // column to solve 
                        &mkl_ldb,     // leading dimension of RHS
                        &mkl_info
                        );

                    assert(mkl_info == 0);

                    for (int k = 0; k < nz; ++k)
                        u(i, j, k) = u_buf[k];
                }
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
                                           * (M_PI * M_PI) * (dt * ns))
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

        double const l2_norm = std::sqrt(sum);

        std::cout
            << "WALLTIME : " << std::setprecision(7) << walltime << " [s]\n"
            << "L2 NORM  : " << std::setprecision(17) << l2_norm << "\n";
    }
};

int main()
{
    feenableexcept(FE_DIVBYZERO);
    feenableexcept(FE_INVALID);
    feenableexcept(FE_OVERFLOW);

    std::cout << "SOLVER   : MKL BATCHED NONCONTIGUOUS Z\n";

    heat_equation_btcs s;

    s.solve();
}
