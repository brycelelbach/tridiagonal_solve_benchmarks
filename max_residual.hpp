///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_F50DCB2B_69E6_4867_8160_5D41D27CCA7D)
#define TSB_F50DCB2B_69E6_4867_8160_5D41D27CCA7D

#include <limits>

#include "assume.hpp"
#include "always_inline.hpp"
#include "fp_equals.hpp"
#include "array3d.hpp"
#include "residual.hpp"

namespace tsb
{

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Layout>
inline T max_residual(
    array3d<T, Layout>& r       // Residual
  , array3d<T, Layout> const& a // Lower band
  , array3d<T, Layout> const& b // Diagonal
  , array3d<T, Layout> const& c // Upper band
  , array3d<T, Layout> const& u // Solution
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename Layout>
inline T max_residual(
    array3d<T, Layout>& r       // Residual
  , array3d<T, Layout> const& a // Lower band
  , array3d<T, Layout> const& b // Diagonal
  , array3d<T, Layout> const& c // Upper band
  , array3d<T, Layout> const& u // Solution
    ) noexcept
{ // {{{
    return max_residual_tile(0, b.ny(), r, a, b, c, u);
} // }}}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T max_residual_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left>& r       // Residual
  , array3d<T, layout_left> const& a // Lower band
  , array3d<T, layout_left> const& b // Diagonal
  , array3d<T, layout_left> const& c // Upper band
  , array3d<T, layout_left> const& u // Solution
    ) noexcept
{ // {{{
    auto const nx = b.nx();
    auto const nz = b.nz();

    residual_tile(j_begin, j_end, r, a, b, c, u);

    T mr = 0.0; 

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    for (auto j = j_begin; j < j_end; ++j)
        for (auto i = 0; i < nx; ++i)
        {
            T min = std::numeric_limits<T>::max();
            T max = std::numeric_limits<T>::min();

            T const* __restrict__ rp = r(i, j, _);

            auto const stride = r.stride_z();

            TSB_ASSUME_ALIGNED_TO_TYPE(rp);

            // NOTE: Strided access.
            #pragma simd
            for (auto k = 0; k < nz; ++k)
            {
                auto const ks = k * stride;

                min = std::min(min, rp[ks]);
                max = std::max(max, rp[ks]);
            }

            T const mr_here = std::max(std::fabs(min), std::fabs(max));

            if ((0 == i) && (0 == j))
                // First iteration, so we have nothing to compare against.
                mr = mr_here;
            else
                // All the columns are the same, so the max residual for
                // each column should be the same.
                TSB_ASSUME(fp_equals(mr, mr_here));
        }

    return mr;
} // }}}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline T max_residual_tile(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& r       // Residual
  , array3d<T, layout_right> const& a // Lower band
  , array3d<T, layout_right> const& b // Diagonal
  , array3d<T, layout_right> const& c // Upper band
  , array3d<T, layout_right> const& u // Solution
    ) noexcept
{ // {{{
    auto const nx = b.nx();
    auto const nz = b.nz();

    residual_tile(j_begin, j_end, r, a, b, c, u);

    T mr = 0.0; 

    TSB_ASSUME(0 == (nz % 16)); // Assume unit stride is divisible by 16.

    for (auto i = 0; i < nx; ++i)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T min = std::numeric_limits<T>::max();
            T max = std::numeric_limits<T>::min();

            T const* rp = r(i, j, _);

            TSB_ASSUME_ALIGNED(rp, 64);

            #pragma simd
            for (auto k = 0; k < nz; ++k)
            {
                min = std::min(min, rp[k]);
                max = std::max(max, rp[k]);
            }

            T const mr_here = std::max(std::fabs(min), std::fabs(max));

            if ((0 == i) && (0 == j))
                // First iteration, so we have nothing to compare against.
                mr = mr_here;
            else
                // All the columns are the same, so the max residual for
                // each column should be the same.
                TSB_ASSUME(fp_equals(mr, mr_here));
        }

    return mr;
} // }}}

} // tsb

#endif // TSB_F50DCB2B_69E6_4867_8160_5D41D27CCA7D
