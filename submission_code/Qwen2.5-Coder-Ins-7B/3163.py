class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        total_sum = 0
        n = len(nums)
        for i in range(n):
            distinct_count = set()
            for j in range(i, n):
                distinct_count.add(nums[j])
                total_sum += len(distinct_count) ** 2
        return total_sum