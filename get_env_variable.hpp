///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_09005064_7296_4E52_9B06_0A6E950E0F61)
#define TSB_09005064_7296_4E52_9B06_0A6E950E0F61

#include <string>
#include <iostream>

#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <climits>

template <typename T>
T get_env_variable(std::string const& var, T default_val); 

template <>
bool get_env_variable(std::string const& var, bool default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    std::uint64_t const r = std::strtoul(env_str.c_str(), &env_str_p_end, 10);

    if ((&env_str.back() != env_str_p_end - 1) || ULONG_MAX == r)
    {
        std::cout << "ERROR: invalid value '" << env_str << "' "
                     "for boolean environment variable '" << var << "'"
                     "\n";
        std::exit(1);
    }

    return bool(r);
}

template <>
std::uint64_t get_env_variable(std::string const& var, std::uint64_t default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    std::uint64_t const r = std::strtoul(env_str.c_str(), &env_str_p_end, 10);

    if ((&env_str.back() != env_str_p_end - 1) || ULONG_MAX == r)
    {
        std::cout << "ERROR: invalid value '" << env_str << "' "
                     "for integer environment variable '" << var << "'"
                     "\n";
        std::exit(1);
    }

    return r;
}

template <>
double get_env_variable(std::string const& var, double default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    double const r = std::strtod(env_str.c_str(), &env_str_p_end);

    if ((&env_str.back() != env_str_p_end - 1) || HUGE_VAL == r)
    {
        std::cout << "ERROR: invalid value '" << env_str << "' "
                     "for floating point environment variable '" << var << "'"
                     "\n";
        std::exit(1);
    }

    return r;
}

template <>
float get_env_variable(std::string const& var, float default_val) 
{
    char const* const env_str_p(std::getenv(var.c_str()));

    if (nullptr == env_str_p)
        return default_val;

    std::string const env_str(env_str_p);

    char* env_str_p_end(nullptr);

    float const r = std::strtof(env_str.c_str(), &env_str_p_end);

    if ((&env_str.back() != env_str_p_end - 1) || HUGE_VALF == r)
    {
        std::cout << "ERROR: invalid value '" << env_str << "' "
                     "for floating point environment variable '" << var << "'"
                     "\n";
        std::exit(1);
    }

    return r;
}

#endif // TSB_09005064_7296_4E52_9B06_0A6E950E0F61

