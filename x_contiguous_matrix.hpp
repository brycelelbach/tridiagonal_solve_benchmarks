///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(CXX_E5D6612B_CFD5_4B71_A0C7_55DA931643A4)
#define CXX_E5D6612B_CFD5_4B71_A0C7_55DA931643A4

#include <utility>

#include "assume.hpp"
#include "array3d.hpp"

template <typename T>
struct x_contiguous_matrix
{
    using array = array3d<T, layout_left>;

    using size_type  = typename array::size_type;
    using value_type = typename array::value_type;

  private:
    array a_; // Subdiagonal band.
    array b_; // Diagonal band.
    array c_; // Superdiagonal band. 

  public:
    void resize(
        size_type array_base_align
      , size_type array_align_step
      , size_type plane_base_align
      , size_type nx, size_type ny, size_type nz
        ) noexcept
    {
        // Allocate storage for the matrix. a and c technically have dimensions
        // of nx * ny * (nz - 1), but we make then nx * ny * nz to simplify the
        // padding math.
        a_.resize(
            array_base_align + 1 * array_align_step
          , nx, ny, nz
          , 0, plane_base_align, 0
        );

        b_.resize(
            array_base_align + 2 * array_align_step
          , nx, ny, nz
          , 0, plane_base_align, 0
        );

        c_.resize(
            array_base_align + 3 * array_align_step
          , nx, ny, nz
          , 0, plane_base_align, 0
        );
    }

    array const& a() const noexcept
    {
        return a_;
    }
    array& a() noexcept
    {
        return a_;
    }

    array const& b() const noexcept
    {
        return b_;
    }
    array& b() noexcept
    {
        return b_;
    }

    array const& c() const noexcept
    {
        return c_;
    }
    array& c() noexcept
    {
        return c_;
    }
};

#endif // CXX_E5D6612B_CFD5_4B71_A0C7_55DA931643A4

