class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                distinct_values = set()
                for num in subarray:
                    distinct_values.add(num)
                distinct_count = len(distinct_values)
                total_sum += distinct_count * distinct_count
        return total_sum