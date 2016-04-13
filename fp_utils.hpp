///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_A983ED3B_3C3B_42AF_8057_C5E91CA3242B)
#define CXX_A983ED3B_3C3B_42AF_8057_C5E91CA3242B

#include <type_traits>
#include <limits>

template <
    typename T
  , typename = typename std::enable_if<std::is_floating_point<T>::value>::type
    >
constexpr bool fp_equals(
    T x, T y, T epsilon = std::numeric_limits<T>::epsilon()
    ) noexcept
{
    return ( ((x + epsilon >= y) && (x - epsilon <= y))
           ? true
           : false);
}


#endif // CXX_A983ED3B_3C3B_42AF_8057_C5E91CA3242B

