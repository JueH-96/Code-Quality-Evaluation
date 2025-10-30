class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        import bisect

        n = len(nums)
        value_to_indices = defaultdict(list)
        for idx, v in enumerate(nums):
            value_to_indices[v].append(idx)

        prefix_sum = [0]*(n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]

        max_sum = float('-inf')
        for i in range(n):
            v = nums[i]
            targets = [v + k, v - k]
            for target_value in targets:
                if target_value in value_to_indices:
                    indices_list = value_to_indices[target_value]
                    pos = bisect.bisect_left(indices_list, i)
                    if pos < len(indices_list):
                        index_j = indices_list[pos]
                        if index_j >= i:
                            sum_subarray = prefix_sum[index_j+1] - prefix_sum[i]
                            if sum_subarray > max_sum:
                                max_sum = sum_subarray

        return max_sum if max_sum != float('-inf') else 0