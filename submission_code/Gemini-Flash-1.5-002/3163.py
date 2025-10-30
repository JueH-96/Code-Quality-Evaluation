class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum_of_squares = 0
        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                distinct_count = len(set(sub_array))
                total_sum_of_squares += distinct_count**2
        return total_sum_of_squares