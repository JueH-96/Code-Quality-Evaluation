class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        nums.sort(reverse=True)
        return sum([i*i for i in nums[:k]]) % (10**9 + 7)