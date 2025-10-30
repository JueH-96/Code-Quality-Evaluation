class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix = [0] * n
        suffix = [0] * n
        distinct_prefix = set()
        distinct_suffix = set()
        for i in range(n):
            distinct_prefix.add(nums[i])
            prefix[i] = len(distinct_prefix)
            distinct_suffix.add(nums[n - i - 1])
            suffix[n - i - 1] = len(distinct_suffix)
        return [prefix[i] - suffix[i + 1] if i < n - 1 else prefix[i] for i in range(n)]