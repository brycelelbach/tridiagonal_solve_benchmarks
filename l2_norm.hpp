///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_5EFF3C90_3CCA_4533_8568_64072592C585)
#define TSB_5EFF3C90_3CCA_4533_8568_64072592C585

#include <cmath>

#include "assume.hpp"
#include "array3d.hpp"
#include "fp_equals.hpp"

namespace tsb {

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Exact>
inline T l2_norm(array3d<T, layout_left> const& u, Exact&& exact) noexcept
{
    auto const nx = u.nx();
    auto const ny = u.ny();
    auto const nz = u.nz();

    T l2 = 0.0;

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    for (auto j = 0; j < ny; ++j)
        for (auto i = 0; i < nx; ++i)
        {
            T sum = 0.0;

            T const* __restrict__ up = u(i, j, _);

            auto const stride = u.stride_z();

            TSB_ASSUME_ALIGNED(up, 8);

            // NOTE: Strided access.
            #pragma simd
            for (auto k = 0; k < nz; ++k)
            {
                auto const ks = k * stride;

                T const abs_term = std::fabs(up[ks] - exact(k));
                sum = sum + abs_term * abs_term;
            }

            T const l2_here = std::sqrt(sum);

            if ((0 == i) && (0 == j))
                // First iteration, so we have nothing to compare against.
                l2 = l2_here;
            else
                // All the columns are the same, so the L2 norm for each
                // column should be the same.
                TSB_ASSUME(fp_equals(l2, l2_here));
        }

    return l2;
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Exact>
inline T l2_norm(array3d<T, layout_right> const& u, Exact&& exact) noexcept
{
    auto const nx = u.nx();
    auto const ny = u.ny();
    auto const nz = u.nz();

    T l2 = 0.0;

    TSB_ASSUME(0 == (nz % 16)); // Assume unit stride is divisible by 16.

    for (auto i = 0; i < nx; ++i)
        for (auto j = 0; j < ny; ++j)
        {
            T sum = 0.0;

            T const* __restrict__ up = u(i, j, _);

            TSB_ASSUME_ALIGNED(up, 64);

            #pragma simd
            for (auto k = 0; k < nz; ++k)
            {
                T const abs_term = std::fabs(up[k] - exact(k));
                sum = sum + abs_term * abs_term;
            }

            T const l2_here = std::sqrt(sum);

            if ((0 == i) && (0 == j))
                // First iteration, so we have nothing to compare against.
                l2 = l2_here;
            else
                // All the columns are the same, so the L2 norm for each
                // column should be the same.
                TSB_ASSUME(fp_equals(l2, l2_here));
        }

    return l2;
}

} // tsb

#endif // TSB_5EFF3C90_3CCA_4533_8568_64072592C585

