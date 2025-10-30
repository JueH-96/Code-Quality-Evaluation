class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_or = [0] * n
        suffix_or = [0] * n

        prefix_or[0] = nums[0]
        for i in range(1, n):
            prefix_or[i] = prefix_or[i - 1] | nums[i]

        suffix_or[n - 1] = nums[n - 1]
        for i in range(n - 2, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | nums[i]

        max_or = 0
        for i in range(n):
            left_or = prefix_or[i - 1] if i > 0 else 0
            right_or = suffix_or[i + 1] if i + 1 < n else 0
            shifted_num = nums[i] << k
            total_or = left_or | shifted_num | right_or
            max_or = max(max_or, total_or)

        return max_or