///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_05BBEF11_9A8F_459D_AD2B_212A813271EC)
#define TSB_05BBEF11_9A8F_459D_AD2B_212A813271EC

#include "assume.hpp"
#include "always_inline.hpp"
#include "array3d.hpp"

namespace tsb { namespace streaming
{

// solve(A, u): A * x = u; u = x

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename F>
inline void pre_elimination_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left>& b                 // Diagonal.
  , F&& f
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename F>
inline void pre_elimination_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left>& b                 // Diagonal.
  , F&& f
    ) noexcept
{
    auto const nx = b.nx();

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    // Pre-Elimination: (j_end - j_begin) * (nx) iterations
    for (auto j = j_begin; j < j_end; ++j)
    {
        T* __restrict__ bbeginp = b(_, j, 0);

        TSB_ASSUME_ALIGNED(bbeginp, 64);

        #pragma simd
        for (auto i = 0; i < nx; ++i)
        {
            f(bbeginp[i]);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename F>
inline void forward_elimination_tile(
    typename array3d<T, layout_left>::size_type jA_begin
  , typename array3d<T, layout_left>::size_type jA_end
  , typename array3d<T, layout_left>::size_type ju_begin
  , typename array3d<T, layout_left>::size_type ju_end
  , array3d<T, layout_left> const& a           // Lower band.
  , array3d<T, layout_left>& b                 // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename F>
inline void forward_elimination_tile(
    typename array3d<T, layout_left>::size_type jA_begin
  , typename array3d<T, layout_left>::size_type jA_end
  , typename array3d<T, layout_left>::size_type ju_begin
  , typename array3d<T, layout_left>::size_type ju_end
  , array3d<T, layout_left> const& a           // Lower band.
  , array3d<T, layout_left>& b                 // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept
{
    auto const nx = u.nx();
    auto const nz = u.nz();

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    TSB_ASSUME(u.nx() == a.nx());
    TSB_ASSUME(u.nz() == a.nz());

    TSB_ASSUME(u.nx() == b.nx());
    TSB_ASSUME(u.nz() == b.nz());

    TSB_ASSUME(u.nx() == c.nx());
    TSB_ASSUME(u.nz() == c.nz());

    // Forward Elimination: (nz - 1) * (j_end - j_begin) * (nx) iterations
    for (auto k = 1; k < nz; ++k)
        for ( typename array3d<T, layout_left>::size_type jA = jA_begin
                                                        , ju = ju_begin
            ; jA < jA_end
            ; ++jA, ++ju
            )
        {
            T const* __restrict__ asub1p = a(_, jA, k - 1);

            T const* __restrict__ bsub1p = b(_, jA, k - 1);
            T* __restrict__       bp     = b(_, jA, k);

            T const* __restrict__ csub1p = c(_, jA, k - 1);

            T const* __restrict__ usub1p = u(_, ju, k - 1);
            T* __restrict__       up     = u(_, ju, k);

            TSB_ASSUME_ALIGNED(asub1p, 64);

            TSB_ASSUME_ALIGNED(bsub1p, 64);
            TSB_ASSUME_ALIGNED(bp,     64);

            TSB_ASSUME_ALIGNED(csub1p, 64);

            TSB_ASSUME_ALIGNED(usub1p, 64);
            TSB_ASSUME_ALIGNED(up,     64);

            #pragma simd
            for (auto i = 0; i < nx; ++i)
            {
                f(asub1p[i], bsub1p[i], bp[i], csub1p[i], usub1p[i], up[i]);
            }
        }
}

template <typename T, typename F>
inline void forward_elimination_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left> const& a           // Lower band.
  , array3d<T, layout_left>& b                 // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename F>
inline void forward_elimination_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left> const& a           // Lower band.
  , array3d<T, layout_left>& b                 // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept
{
    forward_elimination_tile(
        j_begin, j_end, j_begin, j_end, a, b, c, u, std::forward<F>(f)
    );
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename F>
inline void pre_substitution_tile(
    typename array3d<T, layout_left>::size_type jA_begin
  , typename array3d<T, layout_left>::size_type jA_end
  , typename array3d<T, layout_left>::size_type ju_begin
  , typename array3d<T, layout_left>::size_type ju_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename F>
inline void pre_substitution_tile(
    typename array3d<T, layout_left>::size_type jA_begin
  , typename array3d<T, layout_left>::size_type jA_end
  , typename array3d<T, layout_left>::size_type ju_begin
  , typename array3d<T, layout_left>::size_type ju_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept
{
    auto const nx = u.nx();
    auto const nz = u.nz();

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    TSB_ASSUME(u.nx() == b.nx());
    TSB_ASSUME(u.nz() == b.nz());

    // Pre-Substitution: (j_end - j_begin) * (nx) iterations
    for ( typename array3d<T, layout_left>::size_type jA = jA_begin
                                                    , ju = ju_begin
        ; jA < jA_end
        ; ++jA, ++ju
        )
    {
        T const* __restrict__ bendp = b(_, jA, nz - 1);

        T* __restrict__       uendp = u(_, ju, nz - 1);

        TSB_ASSUME_ALIGNED(bendp, 64);

        TSB_ASSUME_ALIGNED(uendp, 64);

        #pragma simd
        for (auto i = 0; i < nx; ++i)
        {
            f(bendp[i], uendp[i]);
        }
    }
}

template <typename T, typename F>
inline void pre_substitution_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename F>
inline void pre_substitution_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept
{
    pre_substitution_tile(
        j_begin, j_end, j_begin, j_end, b, u, std::forward<F>(f)
    );
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename F>
inline void back_substitution_tile(
    typename array3d<T, layout_left>::size_type jA_begin
  , typename array3d<T, layout_left>::size_type jA_end
  , typename array3d<T, layout_left>::size_type ju_begin
  , typename array3d<T, layout_left>::size_type ju_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename F>
inline void back_substitution_tile(
    typename array3d<T, layout_left>::size_type jA_begin
  , typename array3d<T, layout_left>::size_type jA_end
  , typename array3d<T, layout_left>::size_type ju_begin
  , typename array3d<T, layout_left>::size_type ju_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept
{
    auto const nx = u.nx();
    auto const nz = u.nz();

    TSB_ASSUME(0 == (nx % 16)); // Assume unit stride is divisible by 16.

    TSB_ASSUME(u.nx() == b.nx());
    TSB_ASSUME(u.nz() == b.nz());

    TSB_ASSUME(u.nx() == c.nx());
    TSB_ASSUME(u.nz() == c.nz());
 
    // Back Substitution: (nz - 1) * (j_end - j_begin) * (nx) iterations
    for (auto k = nz - 2; k >= 0; --k)
        for ( typename array3d<T, layout_left>::size_type jA = jA_begin
                                                        , ju = ju_begin
            ; jA < jA_end
            ; ++jA, ++ju
            )
        {
            T const* __restrict__ bp     = b(_, jA, k);

            T const* __restrict__ cp     = c(_, jA, k);

            T* __restrict__       up     = u(_, ju, k);
            T const* __restrict__ uadd1p = u(_, ju, k + 1);

            TSB_ASSUME_ALIGNED(bp, 64);

            TSB_ASSUME_ALIGNED(cp, 64);

            TSB_ASSUME_ALIGNED(up, 64);
            TSB_ASSUME_ALIGNED(uadd1p, 64);

            #pragma simd
            for (auto i = 0; i < nx; ++i)
            {
                f(bp[i], cp[i], up[i], uadd1p[i]);
            }
        }
}

template <typename T, typename F>
inline void back_substitution_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename F>
inline void back_substitution_tile(
    typename array3d<T, layout_left>::size_type j_begin
  , typename array3d<T, layout_left>::size_type j_end
  , array3d<T, layout_left> const& b           // Diagonal.
  , array3d<T, layout_left> const& c           // Upper band.
  , array3d<T, layout_left>& u                 // Solution.
  , F&& f
    ) noexcept
{
    back_substitution_tile(
        j_begin, j_end, j_begin, j_end, b, c, u, std::forward<F>(f)
    );
}

///////////////////////////////////////////////////////////////////////////////

namespace repeated_divide {

template <typename T, typename Divider = std::divides<T> >
inline void forward_elimination_kernel(
    T const __restrict__ asub1p
  , T const __restrict__ bsub1p
  , T& __restrict__      bp
  , T const __restrict__ csub1p
  , T const __restrict__ usub1p
  , T& __restrict__      up
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename Divider>
inline void forward_elimination_kernel(
    T const __restrict__ asub1p
  , T const __restrict__ bsub1p
  , T& __restrict__      bp
  , T const __restrict__ csub1p
  , T const __restrict__ usub1p
  , T& __restrict__      up
    ) noexcept
{
    auto constexpr div = Divider();

    // NOTE: a[] starts at a[1], so a[k - 1] is really a[k]
    // T const m0 = a[k - 1] / b[k - 1];
    // b[k] -= m0 * c[k - 1];
    // u[k] -= m0 * u[k - 1];
    T const m0 = div(asub1p, bsub1p);
    T const u0 = up - m0 * usub1p;
    bp -= m0 * csub1p;
    up = u0;
}

template <typename T, typename Divider = std::divides<T> >
inline void pre_substitution_kernel(
    T const __restrict__ bendp
  , T& __restrict__      uendp
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename Divider>
inline void pre_substitution_kernel(
    T const __restrict__ bendp
  , T& __restrict__      uendp
    ) noexcept
{
    auto constexpr div = Divider();

    // u[nz - 1] = u[nz - 1] / b[nz - 1];
    uendp = div(uendp, bendp);
}

template <typename T, typename Divider = std::divides<T> >
inline void back_substitution_kernel(
    T const __restrict__ bp
  , T const __restrict__ cp
  , T& __restrict__      up
  , T const __restrict__ uadd1p
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename Divider>
inline void back_substitution_kernel(
    T const __restrict__ bp
  , T const __restrict__ cp
  , T& __restrict__      up
  , T const __restrict__ uadd1p
    ) noexcept
{
    auto constexpr div = Divider();

    // u[k] = (u[k] - c[k] * u[k + 1]) / b[k];
    up = div((up - cp * uadd1p), bp);
} 

} // repeated_divide

///////////////////////////////////////////////////////////////////////////////

namespace cached_divide {

template <typename T, typename Divider = std::divides<T> >
inline void pre_elimination_kernel(
    T& __restrict__ bbeginp
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename Divider>
inline void pre_elimination_kernel(
    T& __restrict__ bbeginp
    ) noexcept
{
    auto constexpr div = Divider();

    // b^-1[0] = 1.0 / b[0] 
    bbeginp = div(1.0, bbeginp);
}

template <typename T, typename Divider = std::divides<T> >
inline void forward_elimination_kernel(
    T const __restrict__ asub1p
  , T const __restrict__ bsub1p
  , T& __restrict__      bp
  , T const __restrict__ csub1p
  , T const __restrict__ usub1p
  , T& __restrict__      up
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T, typename Divider>
inline void forward_elimination_kernel(
    T const __restrict__ asub1p
  , T const __restrict__ bsub1p
  , T& __restrict__      bp
  , T const __restrict__ csub1p
  , T const __restrict__ usub1p
  , T& __restrict__      up
    ) noexcept
{
    auto constexpr div = Divider();

    // NOTE: a[] starts at a[1], so a[k - 1] is really a[k]
    // T const m0 = a[k - 1] * b^-1[k - 1];
    // b^-1[k] = 1.0 / (b[k] - m0 * c[k - 1]);
    // u[k] = u[k] - m0 * u[k - 1];
    T const m0 = asub1p * bsub1p;
    T const b0 = div(1.0, (bp - m0 * csub1p));
    up -= m0 * usub1p;
    bp = b0;
}

template <typename T>
inline void pre_substitution_kernel(
    T const __restrict__ bendp
  , T& __restrict__      uendp
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void pre_substitution_kernel(
    T const __restrict__ bendp
  , T& __restrict__      uendp
    ) noexcept
{
    // u[nz - 1] = u[nz - 1] * b^-1[nz - 1];
    uendp *= bendp;
}

template <typename T>
inline void back_substitution_kernel(
    T const __restrict__ bp
  , T const __restrict__ cp
  , T& __restrict__      up
  , T const __restrict__ uadd1p
    ) noexcept TSB_ALWAYS_INLINE;

template <typename T>
inline void back_substitution_kernel(
    T const __restrict__ bp
  , T const __restrict__ cp
  , T& __restrict__      up
  , T const __restrict__ uadd1p
    ) noexcept
{
    // u[k] = (u[k] - c[k] * u[k + 1]) * b^-1[k];
    up = (up - cp * uadd1p) * bp;
} 

} // cached_divide

}} // tsb::streaming

#endif // TSB_05BBEF11_9A8F_459D_AD2B_212A813271EC

