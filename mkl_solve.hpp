///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_BE8E5D03_4A96_49EC_BD4F_FD441B706793)
#define TSB_BE8E5D03_4A96_49EC_BD4F_FD441B706793

#include <mkl.h>

#include "assume.hpp"
#include "always_inline.hpp"
#include "array3d.hpp"

namespace tsb { namespace mkl 
{

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void solve(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& a                 // Lower band.
  , array3d<T, layout_right>& b                 // Diagonal.
  , array3d<T, layout_right>& c                 // Upper band.
  , array3d<T, layout_right>& u                 // Solution.
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void solve(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& a                 // Lower band.
  , array3d<T, layout_right>& b                 // Diagonal.
  , array3d<T, layout_right>& c                 // Upper band.
  , array3d<T, layout_right>& u                 // Solution.
    ) noexcept
{
    auto const nx = u.nx();
    auto const nz = u.nz();

    TSB_ASSUME(0 == (nz % 16)); // Assume unit stride is divisible by 16.

    TSB_ASSUME(u.nx() == a.nx());
    TSB_ASSUME(u.ny() == a.ny());
    TSB_ASSUME(u.nz() == a.nz());

    TSB_ASSUME(u.nx() == b.nx());
    TSB_ASSUME(u.ny() == b.ny());
    TSB_ASSUME(u.nz() == b.nz());

    TSB_ASSUME(u.nx() == c.nx());
    TSB_ASSUME(u.ny() == c.ny());
    TSB_ASSUME(u.nz() == c.nz());

    for (auto i = 0; i < nx; ++i)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__ ap = a(i, j, _);

            T* __restrict__ bp = b(i, j, _);

            T* __restrict__ cp = c(i, j, _);

            T* __restrict__ up = u(i, j, _);

            TSB_ASSUME_ALIGNED(ap, 64);

            TSB_ASSUME_ALIGNED(bp, 64);

            TSB_ASSUME_ALIGNED(cp, 64);

            TSB_ASSUME_ALIGNED(up, 64);

            gtsv(nz, 1, ap, bp, cp, up, nz);          
        }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void gtsv(
    MKL_INT const n    // Matrix order.
  , MKL_INT const nrhs // # of RHSes.
  , T* __restrict__ a  // Lower band.
  , T* __restrict__ b  // Diagonal band.
  , T* __restrict__ c  // Upper band.
  , T* __restrict__ u  // RHS.
  , MKL_INT const ldb  // Leading dimension of the RHS.
    ) noexcept TSB_ALWAYS_INLINE;

template <>
inline void gtsv<float>(
    MKL_INT const n       // Matrix order.
  , MKL_INT const nrhs    // # of RHSes.
  , float* __restrict__ a // Lower band.
  , float* __restrict__ b // Diagonal band.
  , float* __restrict__ c // Upper band.
  , float* __restrict__ u // RHS.
  , MKL_INT const ldb     // Leading dimension of the RHS.
    )
{
    MKL_INT info = 0;

    sgtsv_(&n, &nrhs, a, b, c, u, &ldb, &info);

    TSB_ASSUME(0 == info);
}

template <>
inline void gtsv<double>(
    MKL_INT const n        // Matrix order.
  , MKL_INT const nrhs     // # of RHSes.
  , double* __restrict__ a // Lower band.
  , double* __restrict__ b // Diagonal band.
  , double* __restrict__ c // Upper band.
  , double* __restrict__ u // RHS.
  , MKL_INT const ldb      // Leading dimension of the RHS.
    )
{
    MKL_INT info = 0;

    dgtsv_(&n, &nrhs, a, b, c, u, &ldb, &info);

    TSB_ASSUME(0 == info);
}

}} // tsb::mkl

#endif // TSB_BE8E5D03_4A96_49EC_BD4F_FD441B706793

