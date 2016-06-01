///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_CC1C4B22_3289_48FD_AE9B_BB88B3928D01)
#define CXX_CC1C4B22_3289_48FD_AE9B_BB88B3928D01

#include <memory>
#include <type_traits>

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cassert>

template <
    typename T
  , typename = typename std::enable_if<std::is_unsigned<T>::value>::type
    >
inline constexpr bool is_power_of_2(T t)
{
    return bool((t != 0) && !(t & (t - 1)));
}

template <typename T>
struct free_deleter
{
    void operator()(T* p) const noexcept
    {
        std::free(p);
    }
};

template <
    typename T
  , std::uint64_t Alignment = 64
  , typename = typename std::enable_if<is_power_of_2(Alignment)>::type
    >
inline std::unique_ptr<T[], free_deleter<T> > make_aligned_array(
    std::ptrdiff_t size
    ) noexcept
{
    static_assert( true == std::is_pod<T>::value
                 , "T must be POD");
    static_assert( 0 == (Alignment % sizeof(void*))
                 , "Alignment must be a multiple of sizeof(void*)");
    static_assert( true == is_power_of_2(Alignment)
                 , "Alignment must be a power of 2");

    void* p = 0;
    int const r = ::posix_memalign(&p, Alignment, size * sizeof(T));
    assert(0 == r);

    __assume_aligned(p, Alignment);

    std::memset(p, 0, size * sizeof(T));

    free_deleter<T> const d;

    return std::unique_ptr<T[], free_deleter<T> >(reinterpret_cast<T*>(p), d);
}

#endif // CXX_CC1C4B22_3289_48FD_AE9B_BB88B3928D01

