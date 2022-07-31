/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <../MetricType.h>
#include <../gpu/GpuIndicesOptions.h>
#include <../gpu/impl/GpuScalarQuantizer.cuh>
#include <../gpu/utils/DeviceVector.cuh>
#include <../gpu/utils/Tensor.cuh>

namespace faiss {
namespace gpu {

class GpuResources;

void runIVFFlatScan(
        Tensor<float, 2, true>& queries,
        Tensor<int, 2, true>& listIds,
        DeviceVector<void*>& listData,
        DeviceVector<void*>& listIndices,
        IndicesOptions indicesOptions,
        DeviceVector<int>& listLengths,
        int maxListLength,
        int k,
        faiss::MetricType metric,
        bool useResidual,
        Tensor<float, 3, true>& residualBase,
        GpuScalarQuantizer* scalarQ,
        // output
        Tensor<float, 2, true>& outDistances,
        // output
        Tensor<Index::idx_t, 2, true>& outIndices,
        GpuResources* res);

} // namespace gpu
} // namespace faiss
