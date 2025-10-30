class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        sum_total = 0
        n = len(nums)
        for i in range(n):
            distinct = set()
            for j in range(i, n):
                distinct.add(nums[j])
                sum_total += len(distinct) ** 2
        return sum_total