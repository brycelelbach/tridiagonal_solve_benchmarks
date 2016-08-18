////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Bryce Adelstein Lelbach aka wash
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include "heat_equation_btcs.hpp"

///////////////////////////////////////////////////////////////////////////////

int main()
{
    heat_equation_btcs_streaming_cached_divide<
        double
      , tsb::nr_rcp_divide<double, float>
    > s;

    s.run();
}

