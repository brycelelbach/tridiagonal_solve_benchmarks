///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_384D4B17_47A4_4193_A7E6_35EBF37E47CA)
#define CXX_384D4B17_47A4_4193_A7E6_35EBF37E47CA

#include <iomanip>
#include <algorithm>

#include <cmath>
#include <cassert>
#include <cstdio>
#include <cstdint>

#include <omp.h>

#include "assume.hpp"
#include "timers.hpp"
#include "get_env_variable.hpp"
#include "fp_utils.hpp"
#include "array3d.hpp"
#include "x_contiguous_matrix.hpp"

#include "copy.hpp"
#include "max_residual.hpp"
#include "l2_norm.hpp"
#include "build_matrix.hpp"
#include "set_initial_conditions.hpp"
#include "streaming_solve.hpp"
#include "divide.hpp"
#include "nr_rcp_divide.hpp"

template <typename T>
struct solver_traits;

template <
    typename Derived
  , typename Timer = high_resolution_timer
>
struct heat_equation_btcs : enable_fp_exceptions
{
    using matrix = typename solver_traits<Derived>::matrix;
 
    using timer = high_resolution_timer;

    using array  = typename matrix::array; 
    using layout = typename array::layout; 

    using size_type  = typename array::size_type;
    using value_type = typename array::value_type;

  private:
    Derived& derived() noexcept
    {
        return *static_cast<Derived*>(this);
    }

    Derived const& derived() const noexcept
    {
        return *static_cast<Derived const*>(this);
    }

  protected:
    bool const verify; // If true, compute and print the max residual and L2 
                       // norm after each time step.
    bool const header; // If true, print the CSV header when printing results.

    size_type const  nx; // X dimension
    size_type const  ny; // Y dimension
    size_type const  nz; // Z dimension
    size_type const  tw; // Tile width: tile length in the Y dimension. 
    value_type const dz; // Spatial step size, e.g. space between grid points
                         // (COMPUTED).

    size_type const array_base_align;
    size_type const array_align_step;
    size_type const plane_base_align;

    size_type const ns;  // Number of time steps to take.
    value_type const dt; // Time step size.

    static value_type constexpr N = 1.0; // Frequency of sine wave in initial
                                         // conditions.
    value_type const D;                  // Diffusion coefficient.
    value_type const A_coef;             // Matrix coefficient (COMPUTED).

    matrix A; // Tridiagonal Matrix A.
    array u;  // Vector u: problem state.
    array r;  // Vector r: residual.

  private:
    // Allocate the arrays if needed.
    void allocate_arrays()
    { 
        // Allocate storage for the problem state.
        u.resize(array_base_align, nx, ny, nz, 0, plane_base_align, 0);

        // Allocate storage for the matrix.
        A.resize(
            array_base_align, array_align_step, plane_base_align
          , nx, ny, nz
        );

        // Allocate storage for the residual.
        r.resize(array_base_align, nx, ny, nz);
    }

  protected:
    value_type exact_solution(
        size_type k // Z (vertical) coordinate. 
      , size_type s // Time step.
        ) const noexcept
    {
        return std::exp( -D * (N * N)
                       * (M_PI * M_PI) * (dt * s))
             * std::sin(N * M_PI * (dz * k));
    }

    value_type initial_conditions(
        size_type k // Z (vertical) cordinate.
        ) const noexcept
    {
        return std::sin(N * M_PI * (dz * k));
    }

  public:
    heat_equation_btcs() noexcept
      : enable_fp_exceptions()
      , verify(get_env_variable<bool>("verify", false))
      , header(get_env_variable<bool>("header", false))
      , nx(get_env_variable<size_type>("nx", 32))
      , ny(get_env_variable<size_type>("ny", 2048))
      , nz(get_env_variable<size_type>("nz", 32))
      , tw(get_env_variable<size_type>("tw", 16))
      , dz(1.0 / (nz - 1))
      , array_base_align(get_env_variable<size_type>("array_base_align", 1 << 30))
      , array_align_step(get_env_variable<size_type>("array_align_step", 9216))
      , plane_base_align(get_env_variable<size_type>("plane_base_align", 1152))
      , ns(get_env_variable<size_type>("ns", 50))
      , dt(get_env_variable<value_type>("dt", 1.0e-7))
      , D(get_env_variable<value_type>("D", 0.1))
      , A_coef(D * dt / (dz * dz))
      , A()
      , u()
      , r()
    {
        // Get the unit stride dimension.
        auto const nunit = this->*std::conditional<
            std::is_same<layout, layout_left>::value
          , std::integral_constant<
                decltype(&heat_equation_btcs::nx), &heat_equation_btcs::nx
            >
          , std::integral_constant<
                decltype(&heat_equation_btcs::ny), &heat_equation_btcs::ny
            >
        >::type::value;

        // Assume unit stride is divisible by 16.
        TSB_ASSUME(16 <= nunit); 
        TSB_ASSUME(0 == (nunit % 16));

        // Assume the Y dimension is divisible by the tile width.
        TSB_ASSUME(0 == (ny % tw));
    }

    void run()
    {
        allocate_arrays();

        derived().initialize();

        timer t;

        timer::value_type solvertime = 0;

        for (int s = 0; s < ns; ++s)
        {
            if (verify)
                copy(tw, r, u);

            timer st;

            derived().step(s);

            solvertime += st.elapsed();

            derived().post_step(s);

            if (verify)
            {
                value_type const resid = max_residual(r, A.a(), A.b(), A.c(), u);

                value_type const l2 = l2_norm(
                    u 
                  , [=] (size_type k) noexcept
                    { return exact_solution(k, s + 1); }
                );

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

        value_type const l2 = l2_norm(
            u
          , [=] (size_type k) noexcept
            { return exact_solution(k, ns); }
        );

        value_type const bandwidth = ( ( sizeof(value_type)
                                       * ( (4.0 * nx * ny * nz)
                                         + (2.0 * nx * ny * (nz - 1)))
                                       * ns)
                                     / (1 << 30))
                                   / solvertime;

        if (header)
            std::cout <<
                "Benchmark Variant,"
                "Diffusion Coefficient (D),"
                "X (Horizontal) Extent (nx),"
                "Y (Horizontal) Extent (ny),"
                "Z (Vertical) Extent (nz),"
                "Tile Width (tw),"
                "Array Base Align (bytes),"
                "Array Align Step (bytes),"
                "Plane Base Align (bytes),"
                "# of Timesteps (ns),"
                "Timestep Size (dt),"
                "# of Threads,"
                "Wall Time " << t.units() << ","
                "Solver Time " << t.units() << ","
                "Solver Bandwidth (GB/" << t.units() << "),"
                "L2 Norm"
                ;

        std::cout
            << derived().name() << "." << TSB_BUILD_TYPE << ","
            << D << ","
            << nx << ","
            << ny << ","
            << nz << ","
            << tw << ","
            << array_base_align << ","
            << array_align_step << ","
            << plane_base_align << ","
            << ns << ","
            << dt << ","
            << omp_get_max_threads() << ","
            << std::setprecision(7) << walltime << ","
            << std::setprecision(7) << solvertime << ","
            << std::setprecision(7) << bandwidth << ","
            << std::setprecision(17) << l2 << "\n"
            ;
    }
};

template <
    typename T
  , typename Derived
  , typename Timer = high_resolution_timer
    >
struct heat_equation_btcs_streaming
  : heat_equation_btcs<
        heat_equation_btcs_streaming<T, Derived, Timer>
      , Timer
    >
{
    using base_type = heat_equation_btcs<
        heat_equation_btcs_streaming<T, Derived, Timer>
      , Timer
    >;

    using size_type = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    static_assert(
           std::is_same<value_type, double>::value
        || std::is_same<value_type, float>::value
      , "value_type must be either double or float."
    );

  private:
    Derived& derived() noexcept
    {
        return *static_cast<Derived*>(this);
    }

    Derived const& derived() const noexcept
    {
        return *static_cast<Derived const*>(this);
    }

  public:
    void initialize() noexcept
    {
        build_matrix(
            this->tw, this->A_coef, this->A.a(), this->A.b(), this->A.c()
        );

        set_initial_conditions(
            this->tw, this->u
          , [=] (size_type k) noexcept
            { return this->initial_conditions(k); }
        );
    }

    void step(size_type s) noexcept
    {
        derived().step(s);
    }

    void post_step(size_type s) noexcept
    {
        build_matrix(
            this->tw, this->A_coef, this->A.a(), this->A.b(), this->A.c()
        );
    }

    static std::string name() noexcept
    {
        return std::string("STREAMING.")
             + ( std::is_same<value_type, double>::value
               ? "DOUBLE-PRECISION."
               : "SINGLE-PRECISION.")
             + Derived::name();
    }
};

template <typename T, typename Derived, typename Timer>
struct solver_traits<heat_equation_btcs_streaming<T, Derived, Timer> >
{
    using matrix = x_contiguous_matrix<T>;
};

template <
    typename T
  , typename Divider = tsb::divide<T>
  , typename Timer   = high_resolution_timer
    >
struct heat_equation_btcs_streaming_repeated_divide
  : heat_equation_btcs_streaming<
        T
      , heat_equation_btcs_streaming_repeated_divide<T, Divider, Timer>
      , Timer
    >
{
    using base_type = heat_equation_btcs_streaming<
        T
      , heat_equation_btcs_streaming_repeated_divide<T, Divider, Timer>
      , Timer
    >;

    using size_type = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    using divider = Divider;

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            using namespace tsb::streaming;

            forward_elimination(
                j_begin, j_end, this->A.a(), this->A.b(), this->A.c(), this->u
              , repeated_divide::forward_elimination_kernel<value_type, divider>
            );

            pre_substitution(
                j_begin, j_end, this->A.b(), this->u
              , repeated_divide::pre_substitution_kernel<value_type, divider>
            );

            back_substitution(
                j_begin, j_end, this->A.b(), this->A.c(), this->u
              , repeated_divide::back_substitution_kernel<value_type, divider>
            );
        }
    }

    static std::string name() noexcept
    {
        return std::string("REPEATED-") + divider::name();
    }
};

template <
    typename T
  , typename Divider = tsb::divide<T>
  , typename Timer = high_resolution_timer
    >
struct heat_equation_btcs_streaming_cached_divide
  : heat_equation_btcs_streaming<
        T
      , heat_equation_btcs_streaming_cached_divide<T, Divider, Timer>
      , Timer
    >
{
    using base_type = heat_equation_btcs_streaming<
        T
      , heat_equation_btcs_streaming_cached_divide<T, Divider, Timer>
      , Timer
    >;

    using size_type = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    using divider = Divider;

    static_assert(
           std::is_same<value_type, double>::value
        || std::is_same<value_type, float>::value
      , "value_type must be either double or float."
    );

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            using namespace tsb::streaming;

            pre_elimination(
                j_begin, j_end, this->A.b() 
              , cached_divide::pre_elimination_kernel<value_type, divider>
            );

            forward_elimination(
                j_begin, j_end, this->A.a(), this->A.b(), this->A.c(), this->u
              , cached_divide::forward_elimination_kernel<value_type, divider>
            );

            pre_substitution(
                j_begin, j_end, this->A.b(), this->u
              , cached_divide::pre_substitution_kernel<value_type>
            );

            back_substitution(
                j_begin, j_end, this->A.b(), this->A.c(), this->u
              , cached_divide::back_substitution_kernel<value_type>
            );
        }
    }

    static std::string name() noexcept
    {
        return std::string("CACHED-") + divider::name();
    }
};

#endif // CXX_384D4B17_47A4_4193_A7E6_35EBF37E47CA

