///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_E4F4B515_8E1B_4301_8B18_7A802EDAC5D2)
#define TSB_E4F4B515_8E1B_4301_8B18_7A802EDAC5D2

#include <chrono>

#include <cstdint>

namespace tsb {

struct high_resolution_timer
{
    using value_type = double;

    high_resolution_timer()
      : start_time_(take_time_stamp())
    {}

    void restart()
    {
        start_time_ = take_time_stamp();
    }

    value_type elapsed() const // Return elapsed time in seconds.
    {
        return value_type(take_time_stamp() - start_time_) * 1e-9;
    }

    std::uint64_t elapsed_nanoseconds() const
    {
        return take_time_stamp() - start_time_;
    }

    static constexpr char const* units() noexcept
    {
        return "s";
    }

  protected:
    static std::uint64_t take_time_stamp()
    {
        return std::chrono::duration_cast<std::chrono::nanoseconds>
            (std::chrono::steady_clock::now().time_since_epoch()).count();
    }

  private:
    std::uint64_t start_time_;
};

struct tsc_timer
{
    using value_type = std::uint64_t;

    tsc_timer() noexcept
      : start_time_(take_time_stamp())
    {}

    void restart() noexcept
    {
        start_time_ = take_time_stamp();
    }

    value_type elapsed() const noexcept
    {
        return take_time_stamp() - start_time_;
    }

    static constexpr char const* units() noexcept
    {
        return "tsc";
    }

  protected:
    static value_type take_time_stamp() noexcept
    {
        std::uint32_t lo = 0, hi = 0;
        __asm__ __volatile__ (
            "rdtscp ;\n"
            : "=a" (lo), "=d" (hi)
            :
            : "rcx");
        return ((static_cast<value_type>(hi)) << 32) | lo;
    }

  private:
    value_type start_time_;
};

} // tsb

#endif // TSB_E4F4B515_8E1B_4301_8B18_7A802EDAC5D2

