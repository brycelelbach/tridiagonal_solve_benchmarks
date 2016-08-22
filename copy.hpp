///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_E343F9D4_3FF8_4928_8F22_A6720FC1BD34)
#define TSB_E343F9D4_3FF8_4928_8F22_A6720FC1BD34

#include "assume.hpp"
#include "always_inline.hpp"
#include "array3d.hpp"

namespace tsb
{

// copy(dest, src): dest = src 

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename Layout>
inline void copy(
    typename array3d<T, Layout>::size_type tw
  , array3d<T, Layout>& dest
  , array3d<T, Layout> const& src
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename Layout>
inline void copy(
    typename array3d<T, Layout>::size_type tw
  , array3d<T, Layout>& dest
  , array3d<T, Layout> const& src
    ) noexcept
{ // {{{
    auto const ny = src.ny();

    #pragma omp parallel for schedule(static) 
    for (auto j = 0; j < ny; j += tw)
    {
        auto const j_begin = j;
        auto const j_end   = j + tw;

        copy_tile(j_begin, j_end, dest, src);
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void copy_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left>& dest
  , array3d<T, layout_left> const& src
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void copy_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left>& dest
  , array3d<T, layout_left> const& src
    ) noexcept
{ // {{{
    auto const nx = src.nx();
    auto const nz = src.nz();

    TSB_ASSUME(0 == (nx % 16)); 

    TSB_ASSUME(dest.nx() == src.nx());
    TSB_ASSUME(dest.ny() == src.ny());
    TSB_ASSUME(dest.nz() == src.nz());

    for (auto k = 0; k < nz; ++k)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__       destp = dest(_, j, k);
            T const* __restrict__ srcp  = src (_, j, k);

            TSB_ASSUME_ALIGNED(destp, 64);
            TSB_ASSUME_ALIGNED(srcp,  64);

            #pragma simd
            for (auto i = 0; i < nx; ++i)
                destp[i] = srcp[i];
        }
} // }}}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void copy_tile(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& dest
  , array3d<T, layout_right> const& src
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void copy_tile(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& dest
  , array3d<T, layout_right> const& src
    ) noexcept
{ // {{{
    auto const nx = src.nx();
    auto const nz = src.nz();

    TSB_ASSUME(0 == (nz % 16)); 

    TSB_ASSUME(dest.nx() == src.nx());
    TSB_ASSUME(dest.ny() == src.ny());
    TSB_ASSUME(dest.nz() == src.nz());

    for (auto i = 0; i < nx; ++i)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__       destp = dest(i, j, _);
            T const* __restrict__ srcp  = src (i, j, _);

            TSB_ASSUME_ALIGNED(destp, 64);
            TSB_ASSUME_ALIGNED(srcp,  64);

            #pragma simd
            for (auto k = 0; k < nz; ++k)
                destp[k] = srcp[k];
        }
} // }}}

} // tsb

#endif // TSB_E343F9D4_3FF8_4928_8F22_A6720FC1BD34

