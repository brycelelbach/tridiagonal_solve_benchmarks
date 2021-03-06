///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_384D4B17_47A4_4193_A7E6_35EBF37E47CA)
#define TSB_384D4B17_47A4_4193_A7E6_35EBF37E47CA

#include <algorithm>

#include <cmath>
#include <cstdint>

#include <omp.h>

#include "assume.hpp"
#include "timers.hpp"
#include "get_env_variable.hpp"
#include "enable_fp_exceptions.hpp"
#include "array3d.hpp"
#include "full_matrix.hpp"
#include "rolling_matrix.hpp"
#include "printer.hpp"
#include "copy.hpp"
#include "max_residual.hpp"
#include "l2_norm.hpp"
#include "build_matrix.hpp"
#include "set_initial_conditions.hpp"
#include "streaming_solve.hpp"
#include "mkl_solve.hpp"
#include "operator_divider.hpp"
#include "nr_rcp_divider.hpp"
#include "align_policies.hpp"

#warning Update plane_pad -> plane_padding in scripts

namespace tsb
{

template <typename T>
struct solver_traits;

template <typename Derived>
struct heat_equation_btcs : enable_fp_exceptions
{
    using matrix = typename solver_traits<Derived>::matrix;
    using array  = typename matrix::array; 
    using layout = typename array::layout; 
 
    using timer = typename solver_traits<Derived>::timer;

    using size_type  = typename array::size_type;
    using value_type = typename array::value_type;

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

  protected:
    bool const verify; // If true, compute and print the max residual and L2 
                       // norm after each time step.
    bool const header; // If true, print the CSV header when printing results.

    io::output_format_enum const output_format;

    size_type const  nx; // X dimension
    size_type const  ny; // Y dimension
    size_type const  nz; // Z dimension
    size_type const  tw; // Tile width: tile length in the Y dimension. 
    value_type const dz; // Spatial step size, e.g. space between grid points
                         // (COMPUTED).

    size_type const array_base_align;
    size_type const array_align_step;
    size_type const plane_padding;

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
        u.resize(array_base_align, nx, ny, nz, 0, plane_padding, 0);

        // Allocate storage for the matrix.
        A.resize(
            tw, array_base_align, array_align_step, plane_padding
          , nx, ny, nz
        );

        // For the matrix, disable any alignment strategies that are not
        // enabled in our align_policy.
        auto constexpr ap = solver_traits<Derived>::align_policy;

        auto const array_base_align_ = ( ap & use_array_base_align
                                       ? array_base_align
                                       : 64);
        auto const array_align_step_ = ( ap & use_array_align_step
                                       ? array_align_step
                                       : 0);
        auto const plane_padding_        = ( ap & use_plane_padding
                                       ? plane_padding
                                       : 0);

        // Allocate storage for the residual.
        r.resize(array_base_align_, nx, ny, nz);
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
      , output_format(io::output_format_from_string(
            get_env_variable<std::string>("output_format", "bbb")))
      , nx(get_env_variable<size_type>("nx", 32))
      , ny(get_env_variable<size_type>("ny", 2048))
      , nz(get_env_variable<size_type>("nz", 32))
      , tw(get_env_variable<size_type>("tw", 16))
      , dz(1.0 / (nz - 1))
      , array_base_align(get_env_variable<size_type>("array_base_align", 1 << 30))
      , array_align_step(get_env_variable<size_type>("array_align_step", 9216))
      , plane_padding(get_env_variable<size_type>("plane_padding", 1152))
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

    void run() noexcept
    {
        allocate_arrays();

        derived().initialize();

        set_initial_conditions(
            tw, u
          , [=] (size_type k) noexcept { return initial_conditions(k); }
        );

        timer t;

        typename timer::value_type solvertime = 0;

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

        typename timer::value_type const walltime = t.elapsed();

        value_type const l2 = l2_norm(
            u
          , [=] (size_type k) noexcept
            { return exact_solution(k, ns); }
        );

        value_type const bandwidth = ( ( sizeof(value_type)
                                       * ( (4.0 * nx * ny * nz)
                                         + (4.0 * nx * ny * (nz - 1)))
                                       * ns)
                                     / (1 << 30))
                                   / solvertime;

        std::string const bandwidth_units = std::string("GB/") + t.units();

        size_type const problem_size = ( sizeof(value_type)
                                       * ( (2.0 * nx * ny * nz)
                                         + (2.0 * nx * ny * (nz - 1))));

        size_type const tile_size = ( sizeof(value_type)
                                    * ( (2.0 * nx * tw * nz)
                                      + (2.0 * nx * tw * (nz - 1))));

        std::string const name = ( derived().name() 
                                 + ( std::is_same<value_type, double>::value
                                   ? ".DOUBLE-PRECISION"
                                   : ".SINGLE-PRECISION"));

        using namespace io;

        printer print(output_format);

        print
            ( control_variable,     "VAR",   "Benchmark Variant"
            ,                                ""
            , "%s", name.c_str())
            ( control_variable,     "BUILD", "Build Type"
            ,                                ""
            , "%s", TSB_BUILD_TYPE)
            ( control_variable,     "REV",   "Git Revision"
            ,                                ""
            , "%s", TSB_GIT_REVISION)
            ( control_variable,     "D",     "Diffusion Coefficient (D)"
            ,                                ""
            , "%.7g", D)
            ( control_variable,     "NX",    "X (Horizontal) Extent (nx)"
            ,                                "points"
            , "%u", nx) 
            ( independent_variable, "NY",    "Y (Horizontal) Extent (ny)"
            ,                                "points"
            , "%u", ny) 
            ( control_variable,     "NZ",    "Z (Horizontal) Extent (nz)"
            ,                                "points"
            , "%u", nz) 
            ( independent_variable, "PSIZE", "Problem Size"
            ,                                "bytes"
            , "%u", problem_size) 
            ( independent_variable, "TW",    "Tile Width (tw)"
            ,                                "nx*ny planes"
            , "%u", problem_size) 
            ( independent_variable, "TSIZE", "Tile Size"
            ,                                "bytes"
            , "%u", tile_size) 
            ( control_variable,     "BALGN", "Array Base Align"
            ,                                "bytes"
            , "%u", array_base_align) 
            ( control_variable,     "ALGNS", "Array Align Step"
            ,                                "bytes"
            , "%u", array_align_step) 
            ( control_variable,     "PPAD",  "Plane Padding"
            ,                                "bytes"
            , "%u", plane_padding) 
            ( independent_variable, "NS",    "# of Timesteps (ns)"
            ,                                "steps"
            , "%u", ns) 
            ( control_variable,     "DT",    "Timestep Size (dt)"
            ,                                ""
            , "%.7g", dt) 
            ( independent_variable, "PUS",   "# of Processing Units"
            ,                                "PUs"
            , "%u", ::omp_get_max_threads()) 
            ( dependent_variable,   "WTIME", "Total Wall Time"
            ,                                t.units()
            , "%.7g", walltime) 
            ( dependent_variable,   "STIME", "Solver Wall Time"
            ,                                t.units()
            , "%.7g", solvertime) 
            ( dependent_variable,   "BW",    "Solver Bandwidth"
            ,                                bandwidth_units
            , "%.7g", bandwidth) 
            ( independent_variable, "L2",    "L2 Norm"
            ,                                ""
            , "%.17g", l2) 
        ;

        if (header)
            print.output_headers();

        print.output_values();
    }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
struct heat_equation_btcs_full_matrix
  : heat_equation_btcs<
        heat_equation_btcs_full_matrix<Derived>
    >
{ // {
    using base_type = heat_equation_btcs<
        heat_equation_btcs_full_matrix<Derived>
    >;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

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
    void initialize() noexcept {}

    void step(size_type s) noexcept
    {
        derived().step(s);
    }

    void post_step(size_type s) noexcept
    {
        if (this->verify)
            build_matrix(
                this->tw, this->A_coef, this->A.a(), this->A.b(), this->A.c()
            );
    }

    static std::string name() noexcept
    {
        return Derived::name();
    }
};

///////////////////////////////////////////////////////////////////////////////

template <typename Derived>
struct heat_equation_btcs_rolling_matrix
  : heat_equation_btcs<
        heat_equation_btcs_rolling_matrix<Derived>
    >
{
    using base_type = heat_equation_btcs<
        heat_equation_btcs_rolling_matrix<Derived>
    >;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

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
    void initialize() noexcept {}

    void step(size_type s) noexcept
    {
        derived().step(s);
    }

    void post_step(size_type s) noexcept
    {
        if (this->verify)
            for (auto tn = 0; tn < ::omp_get_max_threads(); ++tn)
            {
                build_matrix_tile(
                    0, this->tw, this->A_coef
                  , this->A.a(tn), this->A.b(tn), this->A.c(tn)
                );
            }
    }

    static std::string name() noexcept
    {
        return Derived::name();
    }
};

///////////////////////////////////////////////////////////////////////////////

template <
    typename T
  , typename Divider = operator_divider<T>
  , typename Timer   = high_resolution_timer
    >
struct heat_equation_btcs_full_matrix_streaming_repeated_divide
  : heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_streaming_repeated_divide<
            T, Divider, Timer
        >
    >
{
    using base_type = heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_streaming_repeated_divide<
            T, Divider, Timer
        >
    >;

    using divider = Divider;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            using namespace streaming;

            build_matrix_tile(
                j_begin, j_end, this->A_coef
              , this->A.a(), this->A.b(), this->A.c()
            );

            forward_elimination_tile(
                j_begin, j_end, this->A.a(), this->A.b(), this->A.c(), this->u
              , repeated_divide::forward_elimination_kernel<value_type, divider>
            );

            pre_substitution_tile(
                j_begin, j_end, this->A.b(), this->u
              , repeated_divide::pre_substitution_kernel<value_type, divider>
            );

            back_substitution_tile(
                j_begin, j_end, this->A.b(), this->A.c(), this->u
              , repeated_divide::back_substitution_kernel<value_type, divider>
            );
        }
    }

    static std::string name() noexcept
    {
        return std::string("FULL-MATRIX.STREAMING.REPEATED-")
             + divider::name();
    }
};

template <
    typename T
  , typename Divider
  , typename Timer
    >
struct solver_traits<
    heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_streaming_repeated_divide<
            T, Divider, Timer
        >
    >
> {
    using matrix = full_matrix<T, layout_left>;

    using timer = Timer; 

    static align_policy_enum constexpr align_policy = use_all_align_policies;
};

///////////////////////////////////////////////////////////////////////////////

template <
    typename T
  , typename Divider = operator_divider<T>
  , typename Timer   = high_resolution_timer
    >
struct heat_equation_btcs_full_matrix_streaming_cached_divide
  : heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_streaming_cached_divide<
            T, Divider, Timer
        >
    >
{
    using base_type = heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_streaming_cached_divide<
            T, Divider, Timer
        >
    >;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    using divider = Divider;

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            using namespace streaming;

            build_matrix_tile(
                j_begin, j_end, this->A_coef
              , this->A.a(), this->A.b(), this->A.c()
            );

            pre_elimination_tile(
                j_begin, j_end, this->A.b() 
              , cached_divide::pre_elimination_kernel<value_type, divider>
            );

            forward_elimination_tile(
                j_begin, j_end, this->A.a(), this->A.b(), this->A.c(), this->u
              , cached_divide::forward_elimination_kernel<value_type, divider>
            );

            pre_substitution_tile(
                j_begin, j_end, this->A.b(), this->u
              , cached_divide::pre_substitution_kernel<value_type>
            );

            back_substitution_tile(
                j_begin, j_end, this->A.b(), this->A.c(), this->u
              , cached_divide::back_substitution_kernel<value_type>
            );
        }
    }

    static std::string name() noexcept
    {
        return std::string("FULL-MATRIX.STREAMING.CACHED-")
             + divider::name();
    }
};

template <
    typename T
  , typename Divider
  , typename Timer
    >
struct solver_traits<
    heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_streaming_cached_divide<
            T, Divider, Timer
        >
    >
> {
    using matrix = full_matrix<T, layout_left>;

    using timer = Timer; 

    static align_policy_enum constexpr align_policy = use_all_align_policies;
};

///////////////////////////////////////////////////////////////////////////////

template <
    typename T
  , typename Timer = high_resolution_timer
    >
struct heat_equation_btcs_full_matrix_mkl_z_contiguous
  : heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_mkl_z_contiguous<
            T, Timer
        >
    >
{
    using base_type = heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_mkl_z_contiguous<
            T, Timer
        >
    >;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            using namespace mkl;

            build_matrix_tile(
                j_begin, j_end, this->A_coef
              , this->A.a(), this->A.b(), this->A.c()
            );

            solve_tile(
                j_begin, j_end, this->A.a(), this->A.b(), this->A.c(), this->u
            );
        }
    }

    static std::string name() noexcept
    {
        return "FULL-MATRIX.MKL.Z-CONTIGUOUS"; 
    }
};

template <
    typename T
  , typename Timer
    >
struct solver_traits<
    heat_equation_btcs_full_matrix<
        heat_equation_btcs_full_matrix_mkl_z_contiguous<
            T, Timer
        >
    >
> {
    using matrix = full_matrix<T, layout_right>;

    using timer = Timer; 

    static align_policy_enum constexpr align_policy = use_all_align_policies;
};

///////////////////////////////////////////////////////////////////////////////

template <
    typename T
  , typename Divider = operator_divider<T>
  , typename Timer = high_resolution_timer
    >
struct heat_equation_btcs_rolling_matrix_streaming_repeated_divide
  : heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_streaming_repeated_divide<
            T, Divider, Timer
        >
    >
{
    using base_type = heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_streaming_repeated_divide<
            T, Divider, Timer
        >
    >;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    using divider = Divider;

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            auto const tn = ::omp_get_thread_num();

            using namespace streaming;

            build_matrix_tile(
                0, this->tw, this->A_coef
              , this->A.a(tn), this->A.b(tn), this->A.c(tn)
            );

            forward_elimination_tile(
                0, this->tw, j_begin, j_end
              , this->A.a(tn), this->A.b(tn), this->A.c(tn), this->u
              , repeated_divide::forward_elimination_kernel<value_type, divider>
            );

            pre_substitution_tile(
                0, this->tw, j_begin, j_end
              , this->A.b(tn), this->u
              , repeated_divide::pre_substitution_kernel<value_type, divider>
            );

            back_substitution_tile(
                0, this->tw, j_begin, j_end
              , this->A.b(tn), this->A.c(tn), this->u
              , repeated_divide::back_substitution_kernel<value_type, divider>
            );
        }
    }

    static std::string name() noexcept
    {
        return std::string("ROLLING-MATRIX.STREAMING.CACHED-")
             + divider::name();
    }
};

template <
    typename T
  , typename Divider
  , typename Timer
    >
struct solver_traits<
    heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_streaming_repeated_divide<
            T, Divider, Timer
        >
    >
> {
    using matrix = rolling_matrix<T, layout_left>;

    using timer = Timer; 

    static align_policy_enum constexpr align_policy =
        align_policy_enum(use_array_base_align | use_array_align_step);
};

///////////////////////////////////////////////////////////////////////////////

template <
    typename T
  , typename Divider = operator_divider<T>
  , typename Timer = high_resolution_timer
    >
struct heat_equation_btcs_rolling_matrix_streaming_cached_divide
  : heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_streaming_cached_divide<
            T, Divider, Timer
        >
    >
{
    using base_type = heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_streaming_cached_divide<
            T, Divider, Timer
        >
    >;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    using divider = Divider;

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            auto const tn = ::omp_get_thread_num();

            using namespace streaming;

            build_matrix_tile(
                0, this->tw, this->A_coef
              , this->A.a(tn), this->A.b(tn), this->A.c(tn)
            );

            pre_elimination_tile(
                0, this->tw, this->A.b(tn) 
              , cached_divide::pre_elimination_kernel<value_type, divider>
            );

            forward_elimination_tile(
                0, this->tw, j_begin, j_end
              , this->A.a(tn), this->A.b(tn), this->A.c(tn), this->u
              , cached_divide::forward_elimination_kernel<value_type, divider>
            );

            pre_substitution_tile(
                0, this->tw, j_begin, j_end
              , this->A.b(tn), this->u
              , cached_divide::pre_substitution_kernel<value_type>
            );

            back_substitution_tile(
                0, this->tw, j_begin, j_end
              , this->A.b(tn), this->A.c(tn), this->u
              , cached_divide::back_substitution_kernel<value_type>
            );
        }
    }

    static std::string name() noexcept
    {
        return std::string("FULL-MATRIX.STREAMING.CACHED-")
             + divider::name();
    }
};

template <
    typename T
  , typename Divider
  , typename Timer
    >
struct solver_traits<
    heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_streaming_cached_divide<
            T, Divider, Timer
        >
    >
> {
    using matrix = rolling_matrix<T, layout_left>;

    using timer = Timer; 

    static align_policy_enum constexpr align_policy =
        align_policy_enum(use_array_base_align | use_array_align_step);
};


///////////////////////////////////////////////////////////////////////////////

template <
    typename T
  , typename Timer = high_resolution_timer
    >
struct heat_equation_btcs_rolling_matrix_mkl_z_contiguous
  : heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_mkl_z_contiguous<
            T, Timer
        >
    >
{
    using base_type = heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_mkl_z_contiguous<
            T, Timer
        >
    >;

    using size_type  = typename base_type::size_type;
    using value_type = typename base_type::value_type;

    void step(size_type s) noexcept
    {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < this->ny; j += this->tw)
        {
            auto const j_begin = j;
            auto const j_end   = j + this->tw;

            auto const tn = ::omp_get_thread_num();

            using namespace mkl;

            build_matrix_tile(
                0, this->tw, this->A_coef
              , this->A.a(tn), this->A.b(tn), this->A.c(tn)
            );

            solve_tile(
                0, this->tw, j_begin, j_end
              , this->A.a(tn), this->A.b(tn), this->A.c(tn), this->u
            );
        }
    }

    static std::string name() noexcept
    {
        return "ROLLING-MATRIX.MKL.Z-CONTIGUOUS"; 
    }
};

template <
    typename T
  , typename Timer
    >
struct solver_traits<
    heat_equation_btcs_rolling_matrix<
        heat_equation_btcs_rolling_matrix_mkl_z_contiguous<
            T, Timer
        >
    >
> {
    using matrix = rolling_matrix<T, layout_right>;

    using timer = Timer; 

    static align_policy_enum constexpr align_policy
        = align_policy_enum(use_array_base_align | use_array_align_step);
};

} // tsb

#endif // TSB_384D4B17_47A4_4193_A7E6_35EBF37E47CA

