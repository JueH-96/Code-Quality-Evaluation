import sys 
from typing import List

# Setting a reasonable recursion depth limit, although not strictly needed for iterative DP
# sys.setrecursionlimit(2000) 

# Define the Solution class as required by the problem format
class Solution:
  """
  This class provides a solution to calculate the sum of powers of all subsequences of a given array.
  The power of an array (or subsequence) S is defined as the number of its subsequences T whose sum equals k.
  The problem asks for the sum of powers of all subsequences S of the input array nums.
  The result is returned modulo 10^9 + 7.
  """
  def sumOfPower(self, nums: List[int], k: int) -> int:
    """
    Calculates the sum of power of all subsequences of nums modulo 10^9 + 7.

    The approach uses dynamic programming. Let dp[j][c] be the number of subsequences
    of the elements processed so far that have a sum of j and contain c elements.
    This DP approach is based on the observation that the problem asks for:
    Sum_{S subsequence of nums} (Count of T subsequence of S such that sum(T) = k)
    This sum can be rewritten by changing the order of summation:
    Sum_{T subsequence of nums with sum(T)=k} (Count of S subsequence of nums such that T is subsequence of S)
    A subsequence T of size |T| = c is contained in exactly 2^(n-c) subsequences S of nums.
    So the required sum is: Sum_{T subsequence of nums with sum(T)=k} 2^(n-|T|)
    We can group the subsequences T by their size c:
    Sum_{c=0 to n} (Count of T subsequence of nums with sum(T)=k and size |T|=c) * 2^(n-c)
    
    The DP state dp[j][c] computes exactly the "Count of T subsequence ... with sum(T)=j and size |T|=c".
    So the final answer is Sum_{c=0 to n} dp[k][c] * 2^(n-c) mod (10^9 + 7).

    Args:
        nums: A list of integers.
        k: A positive integer target sum. Constraints state k >= 1.

    Returns:
        The sum of powers modulo 10^9 + 7.
    """
    MOD = 10**9 + 7
    n = len(nums)
    
    # Precompute powers of 2 modulo MOD. powers2[p] stores 2^p % MOD.
    # Needed up to power n. Size n+1 covers indices 0 to n.
    powers2 = [1] * (n + 1)
    for i in range(1, n + 1):
        powers2[i] = (powers2[i-1] * 2) % MOD
        
    # Initialize DP table: dp[j][c] = count of subsequences with sum j and size c.
    # Dimensions: (k+1) x (n+1). Covers sums from 0 to k, sizes from 0 to n.
    # Using lists of lists initialized to 0.
    dp = [[0] * (n + 1) for _ in range(k + 1)]
    
    # Base case: The empty subsequence has sum 0 and size 0. There's 1 such subsequence.
    dp[0][0] = 1  
    
    # Iterate through each number in nums.
    # 'i' is the 0-based index of the current element nums[i]. It also tracks the number of elements considered so far (i elements processed means indices 0 to i-1).
    for i in range(n):
        x = nums[i] # Current number being considered.
        
        # Iterate through possible sums j, from k down to x.
        # We only need to update sums j >= x because we are potentially adding x.
        # The downward iteration for j is crucial for the space-optimized DP update using a single 2D table.
        # It ensures that when we compute dp[j][c] using dp[j-x][c-1], the value dp[j-x][c-1] 
        # comes from the state *before* considering the element x for sum j.
        for j in range(k, x - 1, -1):
            # Iterate through possible sizes c, from i+1 down to 1.
            # After processing element i (which is the (i+1)-th element overall), the maximum possible subsequence size is i+1.
            # The loop correctly covers sizes from 1 up to i+1.
            # The downward iteration for c ensures correctness when updating in-place.
            for c in range(i + 1, 0, -1):
                # Update dp[j][c]: Add the count of subsequences that sum to j-x with size c-1.
                # These represent subsequences formed by taking existing subsequences (from elements nums[0...i-1]) 
                # with sum j-x and size c-1, and appending the current element nums[i] (which is x).
                # The modulo operation ensures the count stays within the required range.
                dp[j][c] = (dp[j][c] + dp[j-x][c-1]) % MOD

    # Calculate the final sum of powers.
    total_power_sum = 0
    # Iterate through all possible sizes c of subsequences summing to k.
    for c in range(n + 1): # Iterate through all possible sizes c from 0 to n
        count = dp[k][c] # Get the number of subsequences with sum k and size c.
        
        # If count is 0, this size contributes nothing to the sum.
        if count > 0: 
            # Each subsequence T of size c with sum k contributes 2^(n-|T|) = 2^(n-c) to the total sum.
            # Multiply the count by the corresponding power of 2, modulo MOD.
            term = (count * powers2[n - c]) % MOD
            # Add this term to the total sum, modulo MOD.
            total_power_sum = (total_power_sum + term) % MOD
            
    # Return the final calculated sum.
    return total_power_sum