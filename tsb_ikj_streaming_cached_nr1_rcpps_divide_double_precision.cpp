////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015 Bryce Adelstein Lelbach aka wash
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include "heat_equation_btcs.hpp"

int main()
{
    tsb::heat_equation_btcs_ikj_streaming_cached_divide<
        double
      , tsb::nr_rcp_divider<double, float, 1>
    > s;

    s.run();
}

