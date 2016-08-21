///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_FB638A80_8CDA_4440_AE85_0B92BC0D73AF)
#define TSB_FB638A80_8CDA_4440_AE85_0B92BC0D73AF

#include <cfenv>

namespace tsb
{

struct enable_fp_exceptions
{
    enable_fp_exceptions() noexcept
    {
        ::feenableexcept(FE_DIVBYZERO);
        ::feenableexcept(FE_INVALID);
        ::feenableexcept(FE_OVERFLOW);
    }
};

} // tsb

#endif // TSB_FB638A80_8CDA_4440_AE85_0B92BC0D73AF

