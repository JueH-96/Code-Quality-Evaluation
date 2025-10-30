class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        count = collections.Counter(nums)
        return sum(n * n * c for n, c in count.items())