class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum_squares = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_set = set()
                for num in subarray:
                    distinct_set.add(num)
                distinct_count = len(distinct_set)
                square_distinct_count = distinct_count ** 2
                total_sum_squares += square_distinct_count
        return total_sum_squares