///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_E5042A14_CF76_4BDC_A563_739E3F3C212E)
#define TSB_E5042A14_CF76_4BDC_A563_739E3F3C212E

#include <omp.h>

#include <vector>

#include "assume.hpp"
#include "array3d.hpp"

namespace tsb
{

template <typename T, typename Layout>
struct rolling_matrix
{
    using array = array3d<T, Layout>;

    using size_type  = typename array::size_type;
    using value_type = typename array::value_type;

  private:
    std::vector<array> a_; // Subdiagonal band.
    std::vector<array> b_; // Diagonal band.
    std::vector<array> c_; // Superdiagonal band. 

  public:
    void resize(
        size_type tw
      , size_type array_base_align
      , size_type array_align_step
      , size_type plane_pad
      , size_type nx, size_type ny, size_type nz
        ) noexcept
    {
        a_.resize(::omp_get_max_threads());
        b_.resize(::omp_get_max_threads());
        c_.resize(::omp_get_max_threads());

        // Allocate storage for the matrix. a and c technically have dimensions
        // of twx * twy * (nz - 1), but we make them twx * twy * nz to simplify
        // the padding math.
        for (auto tn = 0; tn < ::omp_get_max_threads(); ++tn)
        {
            a_[tn].resize(
                array_base_align + 1 * array_align_step
              , nx, tw, nz
              , 0, plane_pad, 0
            );

            b_[tn].resize(
                array_base_align + 2 * array_align_step
              , nx, tw, nz
              , 0, plane_pad, 0
            );

            c_[tn].resize(
                array_base_align + 3 * array_align_step
              , nx, tw, nz
              , 0, plane_pad, 0
            );
        }
    }

    array const& a() const noexcept
    {
        return a(::omp_get_thread_num());
    }
    array& a() noexcept
    {
        return a(::omp_get_thread_num());
    }
    array const& a(size_type tn) const noexcept
    {
        return a_[tn];
    }
    array& a(size_type tn) noexcept
    {
        return a_[tn];
    }

    array const& b() const noexcept
    {
        return b(::omp_get_thread_num());
    }
    array& b() noexcept
    {
        return b(::omp_get_thread_num());
    }
    array const& b(size_type tn) const noexcept
    {
        return b_[tn];
    }
    array& b(size_type tn) noexcept
    {
        return b_[tn];
    }

    array const& c() const noexcept
    {
        return c(::omp_get_thread_num());
    }
    array& c() noexcept
    {
        return c(::omp_get_thread_num());
    }
    array const& c(size_type tn) const noexcept
    {
        return c_[tn];
    }
    array& c(size_type tn) noexcept
    {
        return c_[tn];
    }
};

} // tsb

#endif // TSB_E5042A14_CF76_4BDC_A563_739E3F3C212E

