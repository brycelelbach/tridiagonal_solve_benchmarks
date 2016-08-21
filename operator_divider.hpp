///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_F742275E_B603_4E7A_A115_B2250A8AC4D3)
#define TSB_F742275E_B603_4E7A_A115_B2250A8AC4D3

#include <string>

#include "always_inline.hpp"

namespace tsb {

// Division via C++ operator. The compiler will optimize this to RCP operations
// when possible on platforms which have an RCP instruction for the given type.
template <typename T>
struct operator_divider
{
    T operator()(T num, T den) const noexcept TSB_ALWAYS_INLINE;

    static std::string name() noexcept
    {
        return "DIVIDE";
    }
};

template <typename T>
inline T operator_divider<T>::operator()(
    T num, T den
    ) const noexcept
{
    return num / den;
}

} // tsb

#endif // TSB_5AC3C27C_F3B3_4D90_9630_F0B5807FDFA0

