class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import defaultdict
        from heapq import nlargest

        def get_subarrays(nums, k):
            count = defaultdict(int)
            for i in range(len(nums) - k + 1):
                subarray = tuple(sorted(nums[i:i + k]))
                count[subarray] += 1
            return count

        counts = get_subarrays(nums, k)
        candidates = [(sum(subarray), count) for subarray, count in counts.items() if len(set(subarray)) >= m]
        if not candidates:
            return 0
        max_sum, _ = max(candidates)
        return max_sum