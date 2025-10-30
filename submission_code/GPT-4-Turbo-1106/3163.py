class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        total_sum = 0
        for i in range(len(nums)):
            distinct_values = set()
            for j in range(i, len(nums)):
                distinct_values.add(nums[j])
                total_sum += len(distinct_values) ** 2
        return total_sum