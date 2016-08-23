///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016 Bryce Adelstein Lelbach aka wash <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#if !defined(TSB_2D754184_F5AA_4742_B2D1_7B358BAA6F28)
#define TSB_2D754184_F5AA_4742_B2D1_7B358BAA6F28

#include <cstdio>

#include <memory>
#include <vector>

#include "output_formats.hpp"

namespace tsb { namespace io
{

// Type-erased base class.
struct output_variable_base
{
  protected:
    output_variable_type_enum const vtype_;
    std::string const               tag_;
    std::string const               name_;
    std::string const               units_;
    std::string const               printf_format_;

  public:
    template <
        typename Tag
      , typename Name
      , typename Units
      , typename PrintfFormat
        >
    output_variable_base(     
        output_variable_type_enum vtype
      , Tag&& tag
      , Name&& name
      , Units&& units
      , PrintfFormat&& printf_format
        )
      : vtype_(vtype)
      , tag_(tag)
      , name_(name)
      , units_(units)
      , printf_format_(printf_format)
    {}

    virtual ~output_variable_base() noexcept = default;

    void output_header(output_format_enum of, bool last) const noexcept
    {
        if      (csv_format == of)
        {
            std::string const s = name_
                                + ( units_.empty()
                                  ? std::string("")
                                  : std::string(" [") + units_ + "]")
                                + (last ? "\n" : ",");

            std::printf(s.c_str());
        }
        else if (bbb_format == of)
        {
            std::string const s = std::string("## ")
                                + string_from_output_variable_type(vtype_) + ":"
                                + tag_ + ":"
                                + name_ + ":"
                                + units_ + "\n"; 

            std::printf(s.c_str());
        }

        TSB_ASSUME((csv_format == of) || (bbb_format == of));
    }

    virtual void output_value(output_format_enum, bool) const noexcept = 0;
};

template <typename T>
struct output_variable : output_variable_base
{
  private:
    T value_;

  public:
    template <
        typename Tag
      , typename Name
      , typename Units
      , typename PrintfFormat
      , typename UniversalRefT
        >
    output_variable(     
        output_variable_type_enum vtype
      , Tag&& tag
      , Name&& name
      , Units&& units
      , PrintfFormat&& printf_format
      , UniversalRefT&& value
        )
      : output_variable_base(
            vtype
          , std::forward<Tag>(tag)
          , std::forward<Name>(name)
          , std::forward<Units>(units)
          , std::forward<PrintfFormat>(printf_format)
        ) 
      , value_(value)
    {}

    void output_value(output_format_enum of, bool last) const noexcept
    {
        if      (csv_format == of)
        {
            std::string const s = this->printf_format_
                                + (last ? "\n" : ",");

            std::printf(s.c_str(), value_);
        }
        else if (bbb_format == of)
        {
            std::string const s = this->printf_format_
                                + (last ? "\n" : " ");

            std::printf(s.c_str(), value_);
        }

        TSB_ASSUME((csv_format == of) || (bbb_format == of));
    }
};

struct printer
{
  private:
    output_format_enum const output_format_;
    std::vector<std::unique_ptr<output_variable_base> > variables_;

  public:
    printer(output_format_enum output_format) noexcept
      : output_format_(output_format)
    {}

    template <
        typename Tag
      , typename Name
      , typename Units
      , typename PrintfFormat
      , typename T
        >
    printer& operator()(
        output_variable_type_enum vtype
      , Tag&& tag
      , Name&& name
      , Units&& units
      , PrintfFormat&& printf_format
      , T&& value
        ) noexcept
    {
        variables_.push_back(
            std::unique_ptr<output_variable_base>(
                new output_variable<T>(
                    vtype
                  , std::forward<Tag>(tag)
                  , std::forward<Name>(name)
                  , std::forward<Units>(units)
                  , std::forward<PrintfFormat>(printf_format)
                  , std::forward<T>(value)
                )
            )
        );

        return *this;
    }

    void output_headers() const noexcept
    {
        for (auto i = 0; i < variables_.size() - 1; ++i)
            variables_[i]->output_header(output_format_, false);

        variables_.back()->output_header(output_format_, true);
    }

    void output_values() const noexcept
    {
        for (auto i = 0; i < variables_.size() - 1; ++i)
            variables_[i]->output_value(output_format_, false);

        variables_.back()->output_value(output_format_, true);
    }
};

}} // tsb::io

#endif // TSB_2D754184_F5AA_4742_B2D1_7B358BAA6F28

