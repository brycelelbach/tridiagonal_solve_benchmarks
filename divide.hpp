///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_F742275E_B603_4E7A_A115_B2250A8AC4D3)
#define CXX_F742275E_B603_4E7A_A115_B2250A8AC4D3

#include <string>

namespace tsb {

template <typename T>
struct divide
{
    T operator()(
        T num, T den
        ) const noexcept __attribute__((always_inline));

    static std::string name() noexcept
    {
        return "DIVIDE";
    }
};

template <typename T>
T divide<T>::operator()(
    T num, T den
    ) const noexcept
{
    return num / den;
}

} // tsb

#endif // CXX_5AC3C27C_F3B3_4D90_9630_F0B5807FDFA0

