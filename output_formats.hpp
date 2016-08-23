///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_EE1FA522_13B1_4140_8BDF_83B94C799405)
#define TSB_EE1FA522_13B1_4140_8BDF_83B94C799405

#include <cstdlib>

#include <string>
#include <iostream>

namespace tsb { namespace io
{

enum output_format_enum
{
    invalid_output_format = 0

  , csv_format = 1 // Comma Separate Values format
  , bbb_format = 2 // Bryce's Benchmark Bank (bbb) format 
};

enum output_variable_type_enum
{
    control_variable     = 1
  , independent_variable = 2
  , dependent_variable   = 3
};

inline output_format_enum output_format_from_string(
    std::string const& s
    ) noexcept
{
    if      ("csv" == s)
        return csv_format;
    else if ("bbb" == s)
        return bbb_format;

    std::cout << "ERROR: '" << s << "' is not a supported output format, "
                 "options are 'csv' or 'bbb'"
                 "\n";
    std::exit(1);
    return invalid_output_format;
}

inline char const* string_from_output_variable_type(
    output_variable_type_enum vtype
    ) noexcept
{
    if (control_variable == vtype)
        return "CTL";
    else if (independent_variable == vtype)
        return "IND";
    else if (dependent_variable == vtype)
        return "DEP";

    TSB_ASSUME(  (control_variable == vtype)
              || (independent_variable == vtype)    
              || (dependent_variable == vtype));
    return "";
}

 }} // tsb::io

#endif // TSB_EE1FA522_13B1_4140_8BDF_83B94C799405

