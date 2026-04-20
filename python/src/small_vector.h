// Copyright © 2025 Apple Inc.

#pragma once

#include <limits>
#include <sstream>
#include <type_traits>

#include "mlx/small_vector.h"

#include <nanobind/stl/detail/nb_list.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename Type, size_t Size, typename Alloc>
struct type_caster<mlx::core::SmallVector<Type, Size, Alloc>> {
  using List = mlx::core::SmallVector<Type, Size, Alloc>;
  using Caster = make_caster<Type>;

  // For narrow integer element types (e.g. mx::Shape's int32_t) we fetch each
  // element through a wider integer caster so we can emit a clean ValueError
  // on overflow instead of nanobind's generic "incompatible function
  // arguments" TypeError. See ml-explore/mlx#2681.
  static constexpr bool kNarrowInt = std::is_integral_v<Type> &&
      !std::is_same_v<Type, bool> && (sizeof(Type) < sizeof(long long));

  NB_TYPE_CASTER(
      List,
      const_name("tuple[") + make_caster<Type>::Name + const_name(", ...]"))

  // Not noexcept: on overflow of a narrow integer element we throw
  // nb::value_error so nanobind surfaces a clean ValueError to the user
  // instead of nanobind clobbering our message with a generic
  // "incompatible function arguments" TypeError. See ml-explore/mlx#2681.
  bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) {
    size_t size;
    PyObject* temp;

    // Will initialize 'size' and 'temp'. All return values and
    // return parameters are zero/NULL in the case of a failure.
    PyObject** o = seq_get(src.ptr(), &size, &temp);

    value.clear();
    value.reserve(size);

    bool success = o != nullptr;

    flags = flags_for_local_caster<Type>(flags);

    for (size_t i = 0; i < size; ++i) {
      if constexpr (kNarrowInt) {
        make_caster<long long> wide;
        if (!wide.from_python(o[i], flags, cleanup) ||
            !wide.template can_cast<long long>()) {
          success = false;
          break;
        }
        long long v = wide.operator cast_t<long long>();
        if (v > std::numeric_limits<Type>::max() ||
            v < std::numeric_limits<Type>::min()) {
          std::ostringstream msg;
          msg << "Shape dimension " << v
              << " is outside the supported range ["
              << static_cast<long long>(std::numeric_limits<Type>::min())
              << ", "
              << static_cast<long long>(std::numeric_limits<Type>::max())
              << "]. MLX currently uses 32-bit integers for shape dimensions.";
          Py_XDECREF(temp);
          throw nanobind::value_error(msg.str().c_str());
        }
        value.push_back(static_cast<Type>(v));
      } else {
        Caster caster;
        if (!caster.from_python(o[i], flags, cleanup) ||
            !caster.template can_cast<Type>()) {
          success = false;
          break;
        }
        value.push_back(caster.operator cast_t<Type>());
      }
    }

    Py_XDECREF(temp);

    return success;
  }

  template <typename T>
  static handle from_cpp(T&& src, rv_policy policy, cleanup_list* cleanup) {
    object ret = steal(PyTuple_New(src.size()));

    if (ret.is_valid()) {
      Py_ssize_t index = 0;

      for (auto&& value : src) {
        handle h = Caster::from_cpp(forward_like_<T>(value), policy, cleanup);

        if (!h.is_valid()) {
          ret.reset();
          break;
        }

        NB_TUPLE_SET_ITEM(ret.ptr(), index++, h.ptr());
      }
    }

    return ret.release();
  }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
