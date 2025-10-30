class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        sum_single = 0
        sum_double = 0
        for num in nums:
            if num < 10:
                sum_single += num
            else:
                sum_double += num
        return (2 * sum_single > total_sum) or (2 * sum_double > total_sum)