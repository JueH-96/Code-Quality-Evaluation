class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort(reverse=True)  # Sort in descending order
        top_k = nums[:k]
        sum_squares = sum(num * num for num in top_k) % MOD
        return sum_squares