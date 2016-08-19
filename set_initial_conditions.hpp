///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_FE674664_DBCF_40FD_A5C0_EECCA0FBE78F)
#define TSB_FE674664_DBCF_40FD_A5C0_EECCA0FBE78F

#include "assume.hpp"
#include "array3d.hpp"

namespace tsb {

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename InitialConditions>
void set_initial_conditions_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left>& u // Solution
  , InitialConditions&& ics
    ) noexcept
{
    auto const nx = u.nx();
    auto const nz = u.nz();

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    for (auto k = 0; k < nz; ++k)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__ up = u(_, j, k);

            TSB_ASSUME_ALIGNED(up, 64);

            #pragma simd
            for (auto i = 0; i < nx; ++i)
                up[i] = ics(k);
        }
}

template <typename T, typename InitialConditions>
void set_initial_conditions(
    typename array3d<T, layout_left>::size_type tw
  , array3d<T, layout_left>& u // Solution
  , InitialConditions&& ics
    ) noexcept
{
    auto const ny = u.ny();

    #pragma omp parallel for schedule(static) 
    for (auto j = 0; j < ny; j += tw)
    {
        auto const j_begin = j;
        auto const j_end   = j + tw;

        set_initial_conditions_tile(
            j_begin, j_end, u, std::forward<InitialConditions>(ics)
        );
    }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename InitialConditions>
void set_initial_conditions_tile(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& u // Solution
  , InitialConditions&& ics
    ) noexcept
{
    auto const nx = u.nx();
    auto const nz = u.nz();

    TSB_ASSUME(0 == (nz % 16)); // Assume unit stride is divisible by 16.

    for (auto i = 0; i < nx; ++i)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__ up = u(i, j, _);

            TSB_ASSUME_ALIGNED(up, 64);

            #pragma simd
            for (auto k = 0; k < nz; ++k)
                up[i] = ics(k);
        }
}

template <typename T, typename InitialConditions>
void set_initial_conditions(
    typename array3d<T, layout_right>::size_type tw
  , array3d<T, layout_right>& u // Solution
  , InitialConditions&& ics
    ) noexcept
{
    auto const ny = u.ny();

    #pragma omp parallel for schedule(static) 
    for (auto j = 0; j < ny; j += tw)
    {
        auto const j_begin = j;
        auto const j_end   = j + tw;

        set_initial_conditions_tile(
            j_begin, j_end, u, std::forward<InitialConditions>(ics)
        );
    }
}

} // tsb

#endif // TSB_FE674664_DBCF_40FD_A5C0_EECCA0FBE78F

