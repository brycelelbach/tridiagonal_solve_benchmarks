///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_27F35FCA_6FAA_48D2_8E99_8E7B386841F5)
#define TSB_27F35FCA_6FAA_48D2_8E99_8E7B386841F5

#include "assume.hpp"
#include "always_inline.hpp"
#include "array3d.hpp"

namespace tsb
{

// residual(r, A, u): r = A * u - r

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void residual_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left>& r       // Residual
  , array3d<T, layout_left> const& a // Lower band
  , array3d<T, layout_left> const& b // Diagonal
  , array3d<T, layout_left> const& c // Upper band
  , array3d<T, layout_left> const& u // Solution
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void residual_tile(
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

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    TSB_ASSUME(r.nx() == a.nx());
    TSB_ASSUME(r.nz() == a.nz());

    TSB_ASSUME(r.nx() == b.nx());
    TSB_ASSUME(r.nz() == b.nz());

    TSB_ASSUME(r.nx() == c.nx());
    TSB_ASSUME(r.nz() == c.nz());

    TSB_ASSUME(r.nx() == u.nx());
    TSB_ASSUME(r.nz() == u.nz());

    // First row.
    for (auto j = j_begin; j < j_end; ++j)
    {
        T* __restrict__       r0p = r(_, j, 0);

        T const* __restrict__ b0p = b(_, j, 0);

        T const* __restrict__ c0p = c(_, j, 0);

        T const* __restrict__ u0p = u(_, j, 0);
        T const* __restrict__ u1p = u(_, j, 1);

        TSB_ASSUME_ALIGNED(r0p, 64);

        TSB_ASSUME_ALIGNED(b0p, 64);

        TSB_ASSUME_ALIGNED(c0p, 64);

        TSB_ASSUME_ALIGNED(u0p, 64);
        TSB_ASSUME_ALIGNED(u1p, 64);

        #pragma simd
        for (auto i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed.
            // r[0] = (b[0] * u[0] + c[0] * u[1]) - r[0];
            r0p[i] = (b0p[i] * u0p[i] + c0p[i] * u1p[i]) - r0p[i];
        }
    }

    // Interior rows.
    for (auto k = 1; k < nz - 1; ++k)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__       rp     = r(_, j, k);

            T const* __restrict__ asub1p = a(_, j, k - 1);

            T const* __restrict__ bp     = b(_, j, k);

            T const* __restrict__ cp     = c(_, j, k);

            T const* __restrict__ usub1p = u(_, j, k - 1);
            T const* __restrict__ up     = u(_, j, k);
            T const* __restrict__ uadd1p = u(_, j, k + 1);

            TSB_ASSUME_ALIGNED(rp, 64);

            TSB_ASSUME_ALIGNED(asub1p, 64);

            TSB_ASSUME_ALIGNED(bp, 64);

            TSB_ASSUME_ALIGNED(cp, 64);

            TSB_ASSUME_ALIGNED(usub1p, 64);
            TSB_ASSUME_ALIGNED(up, 64);
            TSB_ASSUME_ALIGNED(uadd1p, 64);

            #pragma simd
            for (auto i = 0; i < nx; ++i)
            {
                // r[k] = ( a[k - 1] * u[k - 1]
                //        + b[k] * u[k]
                //        + c[k] * u[k + 1])
                //      - r[k];
                rp[i] = ( asub1p[i] * usub1p[i]
                        + bp[i] * up[i]
                        + cp[i] * uadd1p[i])
                      - rp[i];
            }
        }

    // Last row.
    for (auto j = j_begin; j < j_end; ++j)
    {
        T* __restrict__       rnz1p = r(_, j, nz - 1);

        T const* __restrict__ anz2p = a(_, j, nz - 2);

        T const* __restrict__ bnz1p = b(_, j, nz - 1);

        T const* __restrict__ unz2p = u(_, j, nz - 2);
        T const* __restrict__ unz1p = u(_, j, nz - 1);

        TSB_ASSUME_ALIGNED(rnz1p, 64);

        TSB_ASSUME_ALIGNED(anz2p, 64);

        TSB_ASSUME_ALIGNED(bnz1p, 64);

        TSB_ASSUME_ALIGNED(unz2p, 64);
        TSB_ASSUME_ALIGNED(unz1p, 64);

        #pragma simd
        for (auto i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed.
            // r[nz - 1] = (a[nz - 2] * u[nz - 2] + b[nz - 1] * u[nz - 1])
            //           - r[nz - 1];
            rnz1p[i] = (anz2p[i] * unz2p[i] + bnz1p[i] * unz1p[i])
                     - rnz1p[i];
        }
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void residual_tile(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& r       // Residual
  , array3d<T, layout_right> const& a // Lower band
  , array3d<T, layout_right> const& b // Diagonal
  , array3d<T, layout_right> const& c // Upper band
  , array3d<T, layout_right> const& u // Solution
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void residual_tile(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , array3d<T, layout_right>& r       // Residual
  , array3d<T, layout_right> const& a // Lower band
  , array3d<T, layout_right> const& b // Diagonal
  , array3d<T, layout_right> const& c // Upper band
  , array3d<T, layout_right> const& u // Solution
    ) noexcept
{ // {{{
    auto const nx = r.nx();
    auto const nz = r.nz();

    TSB_ASSUME(0 == (nz % 16)); // Assume unit stride is divisible by 16.

    TSB_ASSUME(r.nx() == a.nx());
    TSB_ASSUME(r.nz() == a.nz());

    TSB_ASSUME(r.nx() == b.nx());
    TSB_ASSUME(r.nz() == b.nz());

    TSB_ASSUME(r.nx() == c.nx());
    TSB_ASSUME(r.nz() == c.nz());

    TSB_ASSUME(r.nx() == u.nx());
    TSB_ASSUME(r.nz() == u.nz());

    // First row.
    for (auto i = 0; i < nx; ++i)
    {
        T* __restrict__       r0p = r(i, _, 0);

        T const* __restrict__ b0p = b(i, _, 0);

        T const* __restrict__ c0p = c(i, _, 0);

        T const* __restrict__ u0p = u(i, _, 0);
        T const* __restrict__ u1p = u(i, _, 1);

        auto const ac_stride  = a.stride_y();
        auto const bur_stride = b.stride_y();

        TSB_ASSUME_ALIGNED_TO_TYPE(r0p);

        TSB_ASSUME_ALIGNED_TO_TYPE(b0p);

        TSB_ASSUME_ALIGNED_TO_TYPE(c0p);

        TSB_ASSUME_ALIGNED_TO_TYPE(u0p);
        TSB_ASSUME_ALIGNED_TO_TYPE(u1p);

        // NOTE: Strided access.
        #pragma simd
        for (auto j = j_begin; j < j_end; ++j)
        {
            auto const jacs  = j * ac_stride;
            auto const jburs = j * bur_stride;

            // NOTE: The comment is k-indexed. The code is j-indexed.
            // r[0] = (b[0] * u[0] + c[0] * u[1]) - r[0];
            r0p[jburs] = (b0p[jburs] * u0p[jburs] + c0p[jacs] * u1p[jburs])
                       - r0p[jburs];
        }
    }

    // Interior rows.
    for (auto i = 0; i < nx; ++i)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__       rp = r(i, j, _);

            T const* __restrict__ ap = a(i, j, _);

            T const* __restrict__ bp = b(i, j, _);

            T const* __restrict__ cp = c(i, j, _);

            T const* __restrict__ up = u(i, j, _);

            TSB_ASSUME_ALIGNED(rp, 64);

            TSB_ASSUME_ALIGNED(ap, 64);

            TSB_ASSUME_ALIGNED(bp, 64);

            TSB_ASSUME_ALIGNED(cp, 64);

            TSB_ASSUME_ALIGNED(up, 64);

            #pragma simd
            for (auto k = 1; k < nz - 1; ++k)
            {
                // r[k] = ( a[k - 1] * u[k - 1]
                //        + b[k] * u[k]
                //        + c[k] * u[k + 1])
                //      - r[k];
                rp[k] = ( ap[k - 1] * up[k - 1]
                        + bp[k] * up[k]
                        + cp[k] * up[k + 1])
                      - rp[k];
            }
        }

    // Last row.
    for (auto i = 0; i < nx; ++i)
    {
        T* __restrict__       rnz1p = r(i, _, nz - 1);

        T const* __restrict__ anz2p = a(i, _, nz - 2);

        T const* __restrict__ bnz1p = b(i, _, nz - 1);

        T const* __restrict__ unz2p = u(i, _, nz - 2);
        T const* __restrict__ unz1p = u(i, _, nz - 1);

        auto const ac_stride  = a.stride_y();
        auto const bur_stride = b.stride_y();

        TSB_ASSUME_ALIGNED_TO_TYPE(rnz1p);

        TSB_ASSUME_ALIGNED_TO_TYPE(anz2p);

        TSB_ASSUME_ALIGNED_TO_TYPE(bnz1p);

        TSB_ASSUME_ALIGNED_TO_TYPE(unz2p);
        TSB_ASSUME_ALIGNED_TO_TYPE(unz1p);

        // NOTE: Strided access.
        #pragma simd
        for (auto j = j_begin; j < j_end; ++j)
        {
            auto const jacs  = j * ac_stride;
            auto const jburs = j * bur_stride;

            // NOTE: The comment is k-indexed. The code is j-indexed.
            // r[nz - 1] = (a[nz - 2] * u[nz - 2] + b[nz - 1] * u[nz - 1])
            //           - r[nz - 1];
            rnz1p[jburs] = ( anz2p[jacs] * unz2p[jburs]
                           + bnz1p[jburs] * unz1p[jburs])
                         - rnz1p[jburs];
        }
    }
} // }}}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
inline void residual_tile(
    typename array3d<T, layout_ikj>::size_type j_begin
  , typename array3d<T, layout_ikj>::size_type j_end
  , array3d<T, layout_ikj>& r       // Residual
  , array3d<T, layout_ikj> const& a // Lower band
  , array3d<T, layout_ikj> const& b // Diagonal
  , array3d<T, layout_ikj> const& c // Upper band
  , array3d<T, layout_ikj> const& u // Solution
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void residual_tile(
    typename array3d<T, layout_ikj>::size_type j_begin
  , typename array3d<T, layout_ikj>::size_type j_end
  , array3d<T, layout_ikj>& r       // Residual
  , array3d<T, layout_ikj> const& a // Lower band
  , array3d<T, layout_ikj> const& b // Diagonal
  , array3d<T, layout_ikj> const& c // Upper band
  , array3d<T, layout_ikj> const& u // Solution
    ) noexcept
{ // {{{
    auto const nx = b.nx();
    auto const nz = b.nz();

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    TSB_ASSUME(r.nx() == a.nx());
    TSB_ASSUME(r.nz() == a.nz());

    TSB_ASSUME(r.nx() == b.nx());
    TSB_ASSUME(r.nz() == b.nz());

    TSB_ASSUME(r.nx() == c.nx());
    TSB_ASSUME(r.nz() == c.nz());

    TSB_ASSUME(r.nx() == u.nx());
    TSB_ASSUME(r.nz() == u.nz());

    // First row.
    for (auto j = j_begin; j < j_end; ++j)
    {
        T* __restrict__       r0p = r(_, j, 0);

        T const* __restrict__ b0p = b(_, j, 0);

        T const* __restrict__ c0p = c(_, j, 0);

        T const* __restrict__ u0p = u(_, j, 0);
        T const* __restrict__ u1p = u(_, j, 1);

        TSB_ASSUME_ALIGNED(r0p, 64);

        TSB_ASSUME_ALIGNED(b0p, 64);

        TSB_ASSUME_ALIGNED(c0p, 64);

        TSB_ASSUME_ALIGNED(u0p, 64);
        TSB_ASSUME_ALIGNED(u1p, 64);

        #pragma simd
        for (auto i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed.
            // r[0] = (b[0] * u[0] + c[0] * u[1]) - r[0];
            r0p[i] = (b0p[i] * u0p[i] + c0p[i] * u1p[i]) - r0p[i];
        }
    }

    // Interior rows.
    for (auto j = j_begin; j < j_end; ++j)
        for (auto k = 1; k < nz - 1; ++k)
        {
            T* __restrict__       rp     = r(_, j, k);

            T const* __restrict__ asub1p = a(_, j, k - 1);

            T const* __restrict__ bp     = b(_, j, k);

            T const* __restrict__ cp     = c(_, j, k);

            T const* __restrict__ usub1p = u(_, j, k - 1);
            T const* __restrict__ up     = u(_, j, k);
            T const* __restrict__ uadd1p = u(_, j, k + 1);

            TSB_ASSUME_ALIGNED(rp, 64);

            TSB_ASSUME_ALIGNED(asub1p, 64);

            TSB_ASSUME_ALIGNED(bp, 64);

            TSB_ASSUME_ALIGNED(cp, 64);

            TSB_ASSUME_ALIGNED(usub1p, 64);
            TSB_ASSUME_ALIGNED(up, 64);
            TSB_ASSUME_ALIGNED(uadd1p, 64);

            #pragma simd
            for (auto i = 0; i < nx; ++i)
            {
                // r[k] = ( a[k - 1] * u[k - 1]
                //        + b[k] * u[k]
                //        + c[k] * u[k + 1])
                //      - r[k];
                rp[i] = ( asub1p[i] * usub1p[i]
                        + bp[i] * up[i]
                        + cp[i] * uadd1p[i])
                      - rp[i];
            }
        }

    // Last row.
    for (auto j = j_begin; j < j_end; ++j)
    {
        T* __restrict__       rnz1p = r(_, j, nz - 1);

        T const* __restrict__ anz2p = a(_, j, nz - 2);

        T const* __restrict__ bnz1p = b(_, j, nz - 1);

        T const* __restrict__ unz2p = u(_, j, nz - 2);
        T const* __restrict__ unz1p = u(_, j, nz - 1);

        TSB_ASSUME_ALIGNED(rnz1p, 64);

        TSB_ASSUME_ALIGNED(anz2p, 64);

        TSB_ASSUME_ALIGNED(bnz1p, 64);

        TSB_ASSUME_ALIGNED(unz2p, 64);
        TSB_ASSUME_ALIGNED(unz1p, 64);

        #pragma simd
        for (auto i = 0; i < nx; ++i)
        {
            // NOTE: The comment is k-indexed. The code is i-indexed.
            // r[nz - 1] = (a[nz - 2] * u[nz - 2] + b[nz - 1] * u[nz - 1])
            //           - r[nz - 1];
            rnz1p[i] = (anz2p[i] * unz2p[i] + bnz1p[i] * unz1p[i])
                     - rnz1p[i];
        }
    }
} // }}}

} // tsb

#endif // TSB_27F35FCA_6FAA_48D2_8E99_8E7B386841F5

