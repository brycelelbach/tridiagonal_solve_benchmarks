///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_C6134B57_4F3E_4A1D_900E_B93CF6FB93F5)
#define CXX_C6134B57_4F3E_4A1D_900E_B93CF6FB93F5

#include <cstdint>

#if defined(NDEBUG)
    #define TSB_ASSUME(expr)                                                    \
        {                                                                       \
            __assume(expr);                                                     \
        }                                                                       \
        /**/
    #define TSB_ASSUME_ALIGNED(p, alignment)                                    \
        {                                                                       \
            __assume_aligned(p, alignment);                                     \
        }                                                                       \
        /**/
#else
    #define TSB_ASSUME(expr)                                                    \
        {                                                                       \
            assert(expr);                                                       \
            __assume(expr);                                                     \
        }                                                                       \
        /**/
    #define TSB_ASSUME_ALIGNED(p, alignment)                                    \
        {                                                                       \
            assert(0 == (std::uintptr_t(p) % alignment));                       \
            __assume_aligned(p, alignment);                                     \
        }                                                                       \
        /**/
#endif

#endif // CXX_C6134B57_4F3E_4A1D_900E_B93CF6FB93F5

