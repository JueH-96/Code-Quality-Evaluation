class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Since AND operation always produces a result less than or equal to the smallest number
        # and OR operation always produces a result greater than or equal to the largest number,
        # the best strategy is to not perform any operation and just pick the k largest numbers.
        
        nums.sort(reverse=True)  # Sort the numbers in descending order
        max_sum_squares = sum(x**2 for x in nums[:k])  # Sum of squares of the first k elements
        return max_sum_squares % MOD