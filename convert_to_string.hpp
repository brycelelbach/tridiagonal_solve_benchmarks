///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_E26A5551_C432_4FDE_9F59_AA896C2A9E52)
#define TSB_E26A5551_C432_4FDE_9F59_AA896C2A9E52

#include <sstream>

namespace tsb
{

template <typename T>
std::string convert_to_string(T&& t) 
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

} // tsb

#endif // TSB_E26A5551_C432_4FDE_9F59_AA896C2A9E52


