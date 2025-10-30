class Solution:
    def countGoodArrays(self, n: int, m: int, k: int) -> int:
        MOD = 10**9 + 7
        
        # If k is 0, we can only have alternating elements
        if k == 0:
            return m % MOD
        
        # If k is greater than or equal to n, it's impossible to have a good array
        if k >= n:
            return 0
        
        # Calculate the number of good arrays
        # We can choose the first element in m ways
        # For the next k elements, we can choose the same element in 1 way
        # For the remaining (n - k - 1) elements, we can choose any of the m - 1 remaining elements
        # The total number of good arrays can be calculated as:
        
        # Choose the first element (m choices)
        # Choose k pairs (1 choice for each pair)
        # Choose (n - k - 1) remaining elements (m - 1 choices for each)
        
        # Total combinations:
        total_good_arrays = (m * pow(m - 1, n - k - 1, MOD)) % MOD
        
        return total_good_arrays