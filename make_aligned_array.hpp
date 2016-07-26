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

template <std::size_t Alignment>
void* align_ptr(void* ptr, std::ptrdiff_t size, std::ptrdiff_t space) noexcept
{
    if (0 == Alignment)
    {
        if (size > space)
            return nullptr;
        else
            return ptr;
    }

    auto const start = reinterpret_cast<std::uintptr_t>(ptr);

    auto aligned = start; 
    while (0 != (aligned % Alignment))
        ++aligned;

    auto const diff = aligned - start;

    if ((size + diff) > space)
        return nullptr;
    else
    {
        space -= diff;
        return ptr = reinterpret_cast<void*>(aligned);
    }
}

template <typename T>
struct free_deleter
{
    void operator()(T* p) const noexcept
    {
        std::free(p);
    }
};

template <typename T, std::uint64_t Alignment = 64>
struct aligned_array_ptr;

template <
    typename T
  , std::uint64_t Alignment = 64
    >
inline typename std::enable_if<
    is_power_of_2(Alignment)
  , aligned_array_ptr<T, Alignment>
>::type make_aligned_array(std::ptrdiff_t size) noexcept
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

    #if defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700
        // Prior versions of ICPC won't accept a non-type template argument
        // as a parameter to __assume_aligned().
        __assume_aligned(p, Alignment);
    #endif

    std::memset(p, 0, size * sizeof(T));

    return aligned_array_ptr<T, Alignment>(
        reinterpret_cast<T*>(p), reinterpret_cast<T*>(p)
    );
}

template <
    typename T
  , std::uint64_t Alignment = 64
    >
inline typename std::enable_if<
    !is_power_of_2(Alignment)
  , aligned_array_ptr<T, Alignment>
>::type make_aligned_array(std::ptrdiff_t size) noexcept
{
    static_assert( true == std::is_pod<T>::value
                 , "T must be POD");
    static_assert( 0 == (Alignment % sizeof(void*))
                 , "Alignment must be a multiple of sizeof(void*)");

    auto const space = (size * sizeof(T)) + Alignment;

    void* p = ::malloc(space);
    assert(p);

    void* ap = align_ptr<Alignment>(p, size * sizeof(T), space);
    assert(ap);

    #if defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1700
        // Prior versions of ICPC won't accept a non-type template argument
        // as a parameter to __assume_aligned().
        __assume_aligned(ap, Alignment);
    #endif

    std::memset(ap, 0, size * sizeof(T));

    return aligned_array_ptr<T, Alignment>(
        reinterpret_cast<T*>(p), reinterpret_cast<T*>(ap)
    );
}

template <typename T, std::uint64_t Alignment>
struct aligned_array_ptr
{
    static_assert( true == std::is_pod<T>::value
                 , "T must be POD");

    using size_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = typename std::add_pointer<T>::type;
    using reference = typename std::add_lvalue_reference<T>::type;

  private:
    std::unique_ptr<T[], free_deleter<T> > true_ptr_;
    T* aligned_ptr_;

    explicit aligned_array_ptr(T* true_ptr, T* aligned_array_ptr)
      : true_ptr_(true_ptr, free_deleter<T>()), aligned_ptr_(aligned_array_ptr)
    {}

    friend aligned_array_ptr make_aligned_array<T, Alignment>(
        std::ptrdiff_t size
        ) noexcept;

  public:
    aligned_array_ptr()
      : true_ptr_(nullptr, free_deleter<T>()), aligned_ptr_(nullptr)
    {}

    aligned_array_ptr(aligned_array_ptr&& other)
      : true_ptr_(std::move(other.true_ptr_))
      , aligned_ptr_(std::move(other.aligned_ptr_))
    {}

    aligned_array_ptr& operator=(aligned_array_ptr&& other) noexcept
    {
        std::swap(true_ptr_, other.true_ptr_);
        std::swap(aligned_ptr_, other.aligned_ptr_);
    }

    reference operator[](size_type i) const noexcept
    {
        return aligned_ptr_[i];
    }

    reference operator*() const noexcept
    {
        return *aligned_ptr_;
    }

    pointer operator->() const noexcept
    {
        return aligned_ptr_;
    }

    pointer get() const noexcept
    {
        return aligned_ptr_;
    }
};

#endif // CXX_CC1C4B22_3289_48FD_AE9B_BB88B3928D01

