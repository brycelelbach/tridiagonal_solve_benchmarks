///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_DC83EDD6_093D_4373_87A3_1469B32A1CB0)
#define TSB_DC83EDD6_093D_4373_87A3_1469B32A1CB0

#include <cstddef>

#include <string>
#include <type_traits>

#include "always_inline.hpp"
#include "convert_to_string.hpp"

namespace tsb
{

// Newton-Raphson division estimation using fast reciprocal (RCP) instructions.
// The RCPPrecision parameter specifies the precision of the RCP instruction
// used (AVX platforms prior to AVX512 only have a single-precision RCP
// instruction).
template <typename T, typename RCPPrecision, std::ptrdiff_t NRIterations = 1>
struct nr_rcp_divider
{
    static_assert(
           std::is_same<T, double>::value
        || std::is_same<T, float>::value
      , "T must be either double or float."
    );

    static_assert(
           std::is_same<RCPPrecision, double>::value
        || std::is_same<RCPPrecision, float>::value
      , "RCPPrecision must be either double or float."
    );

    static_assert(NRIterations >= 0, "NRIterations cannot be negative.");

    T operator()(T num, T den) const noexcept TSB_ALWAYS_INLINE;

    static std::string name() noexcept
    {
        return std::string("NR")
             + convert_to_string(NRIterations) + "-"
             + ( std::is_same<RCPPrecision, double>::value
               ? "RCPPD-"
               : "RCPPS-")
             + "DIVIDE";
    }
};

template <typename T, typename RCPPrecision, std::ptrdiff_t NRIterations>
T nr_rcp_divider<T, RCPPrecision, NRIterations>::operator()(
    T num, T den
    ) const noexcept
{
    T x = RCPPrecision(1.0) / RCPPrecision(den);

    #pragma unroll
    for (auto i = 0; i < NRIterations; ++i)
        x = x + x * (1.0 - den * x);

    return num * x;
}

} // tsb

#endif // TSB_DC83EDD6_093D_4373_87A3_1469B32A1CB0

