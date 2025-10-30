class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        # Sort the array in descending order
        nums.sort(reverse=True)
        # Sum the squares of the first k elements
        result = sum(x * x for x in nums[:k]) % MOD
        return result