///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_C6134B57_4F3E_4A1D_900E_B93CF6FB93F5)
#define TSB_C6134B57_4F3E_4A1D_900E_B93CF6FB93F5

#include <cstdint>
#include <cassert>

#include "always_inline.hpp"

#if defined(__INTEL_COMPILER)
    #define TSB_BUILTIN_ASSUME(expr)           __assume(expr)
    #define TSB_BUILTIN_ASSUME_ALIGNED(p, agn) __assume_aligned(p, agn)
#else
    #define TSB_BUILTIN_ASSUME(expr)           __builtin_assume(expr)
    #define TSB_BUILTIN_ASSUME_ALIGNED(p, agn) __builtin_assume_aligned(p, agn)
#endif
 
#if defined(NDEBUG)
    #define TSB_ASSUME(expr)                                                   \
        {                                                                      \
            TSB_BUILTIN_ASSUME(expr);                                          \
        }                                                                      \
        /**/
    #define TSB_ASSUME_ALIGNED(p, agn)                                         \
        {                                                                      \
            TSB_BUILTIN_ASSUME_ALIGNED(p, agn);                                \
        }                                                                      \
        /**/
#else
    #define TSB_ASSUME(expr)                                                   \
        {                                                                      \
            assert(expr);                                                      \
            TSB_BUILTIN_ASSUME(expr);                                          \
        }                                                                      \
        /**/
    #define TSB_ASSUME_ALIGNED(p, agn)                                         \
        {                                                                      \
            assert(0 == (std::uintptr_t(p) % agn));                            \
            TSB_BUILTIN_ASSUME_ALIGNED(p, agn);                                \
        }                                                                      \
        /**/
#endif

namespace tsb
{

template <typename T>
void assume_aligned_to_type(T* ptr) noexcept TSB_ALWAYS_INLINE;

// Prior versions of ICPC won't accept a non-type template argument or a
// constexpr expression that is dependent on a template argument as a parameter
// to __assume_aligned(). As a workaround, we can fully specialize
// assume_aligned_to_type() for all the types we care about.
#if !defined(__INTEL_COMPILER) || (__INTEL_COMPILER >= 1700)
    template <typename T>
    void assume_aligned_to_type(T* ptr) noexcept
    {
        TSB_BUILTIN_ASSUME_ALIGNED(ptr, sizeof(T));
    }
#else
    template <>
    void assume_aligned_to_type(float* ptr) noexcept
    {
        TSB_BUILTIN_ASSUME_ALIGNED(ptr, sizeof(float));
    }

    template <>
    void assume_aligned_to_type(double* ptr) noexcept
    {
        TSB_BUILTIN_ASSUME_ALIGNED(ptr, sizeof(double));
    }

    template <>
    void assume_aligned_to_type(float const* ptr) noexcept
    {
        TSB_BUILTIN_ASSUME_ALIGNED(ptr, sizeof(float));
    }

    template <>
    void assume_aligned_to_type(double const* ptr) noexcept
    {
        TSB_BUILTIN_ASSUME_ALIGNED(ptr, sizeof(double));
    }
#endif

} // tsb

#if defined(NDEBUG)
    #define TSB_ASSUME_ALIGNED_TO_TYPE(p)                                       \
        {                                                                       \
            ::tsb::assume_aligned_to_type(p);                                   \
        }                                                                       \
        /**/
#else
    #define TSB_ASSUME_ALIGNED_TO_TYPE(p)                                       \
        {                                                                       \
            assert(0 == (std::uintptr_t(p) % sizeof(decltype(*p))));            \
            ::tsb::assume_aligned_to_type(p);                                   \
        }                                                                       \
        /**/
#endif

#endif // TSB_C6134B57_4F3E_4A1D_900E_B93CF6FB93F5

