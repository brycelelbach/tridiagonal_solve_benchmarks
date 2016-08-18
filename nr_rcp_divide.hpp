///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_5AC3C27C_F3B3_4D90_9630_F0B5807FDFA0)
#define CXX_5AC3C27C_F3B3_4D90_9630_F0B5807FDFA0

#include <cstdint>

namespace tsb {

// Newton-Raphson division estimation using fast reciprocal (RCP) instructions.
// The RCPPrecision parameter specifies the precision of the RCP instruction
// used (AVX platforms prior to AVX512 only have a single-precision RCP
// instruction).
template <typename T, typename RCPPrecision, std::ptrdiff_t NRIterations = 1>
struct nr_rcp_divide
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

    T operator()(
        T num, T den
        ) const noexcept __attribute__((always_inline));

    static std::string name() noexcept
    {
        return std::string("NR-")
             + ( std::is_same<RCPPrecision, double>::value
               ? "RCPPD-"
               : "RCPPS-")
             + "DIVIDE";
    }
};

template <typename T, typename RCPPrecision, std::ptrdiff_t NRIterations>
T nr_rcp_divide<T, RCPPrecision, NRIterations>::operator()(
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

#endif // CXX_5AC3C27C_F3B3_4D90_9630_F0B5807FDFA0

