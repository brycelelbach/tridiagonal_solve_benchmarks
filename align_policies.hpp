///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_3BA10630_D3D5_4BC2_90EF_BEFF874B1B87)
#define TSB_3BA10630_D3D5_4BC2_90EF_BEFF874B1B87

#include "binary_literals.hpp"

namespace tsb {

enum align_policy_enum
{
    use_array_base_align = TSB_BINARY(0001)
  , use_array_align_step = TSB_BINARY(0010)
  , use_plane_pad        = TSB_BINARY(0100)

  , use_all_align_policies = use_array_base_align
                           | use_array_align_step
                           | use_plane_pad
};

} // tsb

#endif // TSB_3BA10630_D3D5_4BC2_90EF_BEFF874B1B87

