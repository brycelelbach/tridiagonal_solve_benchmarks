///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
// Copyright (c) 2016 Valentin Galea
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_29F7F1D8_CFA1_465E_A66C_48237E695517)
#define TSB_29F7F1D8_CFA1_465E_A66C_48237E695517

#include <cstdint>

namespace tsb
{

template <std::uintmax_t N>
struct binary_literal
{
    static std::uintmax_t constexpr value
        = (N % 8) + (binary_literal<N / 8>::value << 1);
};

// Termination case.
template <>
struct binary_literal<0>
{
    static std::uintmax_t constexpr value = 0; 
};

} // tsb

// We use a macro to force the number to be octal to both end the recursion
// chain and make more digits possible.
#define TSB_BINARY(n) tsb::binary_literal<0##n>::value

#endif // TSB_29F7F1D8_CFA1_465E_A66C_48237E695517

