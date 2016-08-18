///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_4D7778C5_461A_412C_81EA_40CA5280ABB5)
#define CXX_4D7778C5_461A_412C_81EA_40CA5280ABB5

#include "assume.hpp"
#include "array3d.hpp"

///////////////////////////////////////////////////////////////////////////////

template <typename T>
void build_matrix_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , T A_coef
  , array3d<T, layout_left>& a // Lower band
  , array3d<T, layout_left>& b // Diagonal
  , array3d<T, layout_left>& c // Upper band
    ) noexcept
{
    auto const nx = b.nx();
    auto const nz = b.nz();

    T const ac_term = -A_coef;
    T const b_term  = 1.0 + 2.0 * A_coef;

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    for (auto k = 0; k < nz; ++k)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__ bp = b(_, j, k);

            TSB_ASSUME_ALIGNED(bp, 64);

            #pragma simd
            for (auto i = 0; i < nx; ++i)
                bp[i] = b_term;
        }

    for (auto k = 0; k < nz - 1; ++k)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* __restrict__ ap = a(_, j, k);
            T* __restrict__ cp = c(_, j, k);

            TSB_ASSUME_ALIGNED(ap, 64);
            TSB_ASSUME_ALIGNED(cp, 64);

            #pragma simd
            for (int i = 0; i < nx; ++i)
            {
                ap[i] = ac_term;
                cp[i] = ac_term;
            }
        }

    // Boundary conditions.
    for (auto j = j_begin; j < j_end; ++j)
    {
        T* __restrict__ bbeginp = b(_, j, 0);
        T* __restrict__ cbeginp = c(_, j, 0);

        T* __restrict__ aendp   = a(_, j, nz - 2);
        T* __restrict__ bendp   = b(_, j, nz - 1);

        __assume_aligned(bbeginp, 64);
        __assume_aligned(cbeginp, 64);

        __assume_aligned(aendp, 64);
        __assume_aligned(bendp, 64);

        #pragma simd
        for (auto i = 0; i < nx; ++i)
        {
            bbeginp[i] = 1.0; cbeginp[i] = 0.0;
            aendp[i]   = 0.0; bendp[i]   = 1.0;
        }
    }
}

template <typename T>
void build_matrix(
    typename array3d<T, layout_left>::size_type tw
  , T A_coef
  , array3d<T, layout_left>& a // Lower band
  , array3d<T, layout_left>& b // Diagonal
  , array3d<T, layout_left>& c // Upper band
    ) noexcept
{
    auto const ny = b.ny();

    #pragma omp parallel for schedule(static) 
    for (auto j = 0; j < ny; j += tw)
    {
        auto const j_begin = j;
        auto const j_end   = j + tw;

        build_matrix_tile(j_begin, j_end, A_coef, a, b, c);
    }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T>
void build_matrix_tile(
    typename array3d<T, layout_right>::size_type j_begin
  , typename array3d<T, layout_right>::size_type j_end
  , T A_coef
  , array3d<T, layout_right>& a // Lower band
  , array3d<T, layout_right>& b // Diagonal
  , array3d<T, layout_right>& c // Upper band
    ) noexcept
{
    auto const nx = b.nx();
    auto const nz = b.nz();

    T const ac_term = -A_coef;
    T const b_term  = 1.0 + 2.0 * A_coef;

    TSB_ASSUME(0 == (nz % 16)); 

    for (auto i = 0; i < nx; ++i)
        for (auto j = j_begin; j < j_end; ++j)
        {
            T* bp = b(i, j, _);

            TSB_ASSUME_ALIGNED(bp, 64);

            #pragma simd
            for (auto k = 0; k < nz; ++k)
                bp[k] = b_term;

            T* ap = a(i, j, _);
            T* cp = c(i, j, _);

            TSB_ASSUME_ALIGNED(ap, 64);
            TSB_ASSUME_ALIGNED(cp, 64);

            #pragma simd
            for (auto k = 0; k < nz - 1; ++k)
            {
                ap[k] = ac_term;
                cp[k] = ac_term;
            }
        }

    // Boundary conditions.
    for (auto i = 0; i < nx; ++i)
    {
        T* bbeginp = b(i, _, 0);
        T* cbeginp = c(i, _, 0);

        T* aendp   = a(i, _, nz - 2);
        T* bendp   = b(i, _, nz - 1);

        auto ac_stride = a.stride_y();
        auto b_stride  = b.stride_y();

        TSB_ASSUME_ALIGNED(bbeginp, 64);
        TSB_ASSUME_ALIGNED(cbeginp, 64);

        TSB_ASSUME_ALIGNED(aendp, 64);
        TSB_ASSUME_ALIGNED(bendp, 64);

        // NOTE: Strided access.
        #pragma simd
        for (auto j = j_begin; j < j_end; ++j)
        {
            auto jacs = j * ac_stride;
            auto jbs  = j * b_stride;

            bbeginp[jbs] = 1.0; cbeginp[jacs] = 0.0;
            aendp[jacs]  = 0.0; bendp[jbs]    = 1.0;
        }
    }
}

template <typename T>
void build_matrix(
    typename array3d<T, layout_right>::size_type tw
  , T A_coef
  , array3d<T, layout_right>& a // Lower band
  , array3d<T, layout_right>& b // Diagonal
  , array3d<T, layout_right>& c // Upper band
    ) noexcept
{
    auto const ny = b.ny();

    #pragma omp parallel for schedule(static) 
    for (auto j = 0; j < ny; j += tw)
    {
        auto const j_begin = j;
        auto const j_end   = j + tw;

        build_matrix_tile(j_begin, j_end, A_coef, a, b, c);
    }
}

#endif // CXX_4D7778C5_461A_412C_81EA_40CA5280ABB5

