class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        dp_even = float('-inf')
        dp_odd = float('-inf')
        if nums[0] % 2 == 0:
            dp_even = nums[0]
        else:
            dp_odd = nums[0]
        for num in nums[1:]:
            if num % 2 == 0:
                new_dp_even = max(dp_even + num, dp_odd + num - x)
                dp_even = new_dp_even
            else:
                new_dp_odd = max(dp_odd + num, dp_even + num - x)
                dp_odd = new_dp_odd
        return max(dp_even, dp_odd)