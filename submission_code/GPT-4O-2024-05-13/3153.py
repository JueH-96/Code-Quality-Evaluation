class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Sort the array in descending order
        nums.sort(reverse=True)
        
        # Take the first k elements
        top_k_elements = nums[:k]
        
        # Calculate the sum of their squares
        sum_of_squares = sum(x * x for x in top_k_elements) % MOD
        
        return sum_of_squares