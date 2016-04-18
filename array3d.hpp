///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_6594A0DA_8E9E_4B31_A32F_38EFDF13289E)
#define CXX_6594A0DA_8E9E_4B31_A32F_38EFDF13289E

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>

struct placeholder {};

constexpr placeholder _ {};

struct layout_left
{
    typedef std::ptrdiff_t size_type;

  private:
    size_type nx_, ny_, nz_;

  public:
    constexpr layout_left() noexcept : nx_(0), ny_(0), nz_(0) {}

    constexpr layout_left(size_type nx, size_type ny, size_type nz) noexcept
      : nx_(nx), ny_(ny), nz_(nz)
    {}

    constexpr size_type operator()(
        size_type i, size_type j, size_type k
        ) const noexcept
    {
        return i + nx_ * j + nx_ * ny_ * k;
    }
    constexpr size_type operator()(
        placeholder, size_type j, size_type k
        ) const noexcept
    {
        return nx_ * j + nx_ * ny_ * k;
    }
    constexpr size_type operator()(
        size_type i, placeholder, size_type k
        ) const noexcept
    {
        return i + nx_ * ny_ * k;
    }
    constexpr size_type operator()(
        size_type i, size_type j, placeholder
        ) const noexcept
    {
        return i + nx_ * j;
    }    

    constexpr size_type stride_x() const noexcept
    {
        return 1;
    }
    constexpr size_type stride_y() const noexcept
    {
        return nx_;
    }
    constexpr size_type stride_z() const noexcept
    {
        return nx_ * ny_;
    }

    constexpr size_type nx() const noexcept
    {
        return nx_;
    }
    constexpr size_type ny() const noexcept
    {
        return ny_;
    }
    constexpr size_type nz() const noexcept
    {
        return nz_;
    }
};

struct layout_right
{
    typedef std::ptrdiff_t size_type;

  private:
    size_type nx_, ny_, nz_;

  public:
    constexpr layout_right() noexcept : nx_(0), ny_(0), nz_(0) {}

    constexpr layout_right(size_type nx, size_type ny, size_type nz) noexcept
      : nx_(nx), ny_(ny), nz_(nz)
    {}

    constexpr size_type operator()(
        size_type i, size_type j, size_type k
        ) const noexcept
    {
        return nz_ * ny_ * i + nz_ * j + k;
    }
    constexpr size_type operator()(
        placeholder, size_type j, size_type k
        ) const noexcept
    {
        return nz_ * j + k;
    }
    constexpr size_type operator()(
        size_type i, placeholder, size_type k
        ) const noexcept
    {
        return nz_ * ny_ * i + k;
    }
    constexpr size_type operator()(
        size_type i, size_type j, placeholder
        ) const noexcept
    {
        return nz_ * ny_ * i + nz_ * j;
    }

    constexpr size_type stride_x() const noexcept
    {
        return nz_ * ny_;
    }
    constexpr size_type stride_y() const noexcept
    {
        return nz_;
    }
    constexpr size_type stride_z() const noexcept
    {
        return 1;
    }

    constexpr size_type nx() const noexcept
    {
        return nx_;
    }
    constexpr size_type ny() const noexcept
    {
        return ny_;
    }
    constexpr size_type nz() const noexcept
    {
        return nz_;
    }
};

template <
    typename T
  , typename Layout
  , std::uint64_t Alignment = 64
    >
struct array3d
{
    typedef typename Layout::size_type size_type;
    typedef T value_type;

  private:
    T* data_;
    Layout layout_;

  public:
    constexpr array3d() noexcept : data_(), layout_() {}

    array3d(size_type nx, size_type ny, size_type nz) noexcept
      : layout_(nx, ny, nz)
    {
        resize(nx(), ny(), nz());
    }

    ~array3d()
    {
        clear();
    }

    void resize(size_type nx, size_type ny, size_type nz) noexcept
    {
        clear();

        assert(0 == ((nx * ny * nz * sizeof(T)) % Alignment));

        void* p = 0; 
        int const r = posix_memalign(&p, Alignment, nx * ny * nz * sizeof(T));
        assert(0 == r);

        std::memset(p, 0, nx * ny * nz * sizeof(T));

        data_   = reinterpret_cast<T*>(p);
        layout_ = Layout(nx, ny, nz);
    }

    void clear() noexcept
    {
        if (data_)
        {
            assert(0 != nx() * ny() * nz());
            std::free(data_);
        }

        data_   = 0;
        layout_ = Layout();
    }

    T* data() const noexcept
    {
        return data_;
    }
    T* data() noexcept
    {
        return data_;
    }

    T const& operator()(size_type i, size_type j, size_type k) const noexcept
    {
        return data_[layout_(i, j, k)];
    }
    T& operator()(size_type i, size_type j, size_type k) noexcept
    {
        return data_[layout_(i, j, k)];
    }

    T const* operator()(placeholder p, size_type j, size_type k) const noexcept
    {
        return &data_[layout_(p, j, k)];
    }
    T* operator()(placeholder p, size_type j, size_type k) noexcept
    {
        return &data_[layout_(p, j, k)];
    }

    T const* operator()(size_type i, placeholder p, size_type k) const noexcept
    {
        return &data_[layout_(i, p, k)];
    }
    T* operator()(size_type i, placeholder p, size_type k) noexcept
    {
        return &data_[layout_(i, p, k)];
    }

    T const* operator()(size_type i, size_type j, placeholder p) const noexcept
    {
        return &data_[layout_(i, j, p)];
    }
    T* operator()(size_type i, size_type j, placeholder p) noexcept
    {
        return &data_[layout_(i, j, p)];
    }

    constexpr size_type stride_x() const noexcept
    {
        return layout_.stride_x();
    }
    constexpr size_type stride_y() const noexcept
    {
        return layout_.stride_y();
    }
    constexpr size_type stride_z() const noexcept
    {
        return layout_.stride_z();
    }

    constexpr size_type nx() const noexcept
    {
        return layout_.nx();
    }
    constexpr size_type ny() const noexcept
    {
        return layout_.ny();
    }
    constexpr size_type nz() const noexcept
    {
        return layout_.nz();
    }
};

#endif // CXX_6594A0DA_8E9E_4B31_A32F_38EFDF13289E

