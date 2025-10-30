import math
from typing import List

class Solution:
    def __init__(self):
        """
        Initializes the Solution object with a memoization table for F(4^k - 1).
        The memoization table `memo_F_pow4` stores the computed values of F(4^k - 1)
        to avoid redundant calculations. Keys are 'k', values are F(4^k - 1).
        """
        self.memo_F_pow4 = {}

    def compute_F_pow4_minus_1(self, k: int) -> int:
        """
        Computes F(4^k - 1) = sum_{x=1}^{4^k - 1} steps(x) using recursion with memoization.
        steps(x) is defined as the number of operations to reduce x to 0 by repeatedly
        applying floor(x/4). This is equivalent to the number of digits in the base 4 
        representation of x for x > 0, and steps(0) = 0.
        F(N) is the prefix sum: F(N) = sum_{x=1}^{N} steps(x).
        This function calculates F up to the largest number representable with k digits in base 4.
        """
        if k == 0:
            # Base case: F(4^0 - 1) = F(0) = 0. There are no positive integers up to 0.
            return 0 
        if k in self.memo_F_pow4:
            # If the value for k is already computed, return it from the cache.
            return self.memo_F_pow4[k]
        
        # Calculate the contribution for numbers with exactly k digits in base 4.
        # These numbers range from 4^(k-1) to 4^k - 1.
        # For each such number x, steps(x) = k.
        pow4_k = pow(4, k)
        # Calculate 4^(k-1) efficiently using integer division.
        pow4_k_minus_1 = pow4_k // 4 
        
        # The count of numbers with exactly k digits is 4^k - 4^(k-1).
        count_k_digits = pow4_k - pow4_k_minus_1
        # The total steps contributed by these numbers is count_k_digits * k.
        term_k = count_k_digits * k
        
        # Recursively compute F(4^(k-1) - 1) which sums steps for numbers with < k digits.
        # The total F(4^k - 1) is the sum for < k digits plus the sum for exactly k digits.
        res = self.compute_F_pow4_minus_1(k-1) + term_k
        
        # Store the computed result in the memoization table before returning.
        self.memo_F_pow4[k] = res
        return res

    def compute_F(self, N: int) -> int:
        """
        Computes F(N) = sum_{x=1}^{N} steps(x).
        This function efficiently calculates the prefix sum of steps up to N.
        It utilizes the formula: F(N) = F(4^k - 1) + (N - 4^k + 1) * (k+1),
        where k is the largest integer such that 4^k <= N. This k means that
        numbers in the range [4^k, N] have k+1 digits in base 4.
        """
        if N == 0:
             # F(0) = 0 by definition.
            return 0
        
        # Find the largest integer k such that 4^k <= N.
        k = 0
        # Initialize pow4_k to 4^0 = 1. This variable will eventually hold 4^k.
        pow4_k = 1 
        
        # Use temporary variables to find k and 4^k. This preserves initial values if needed.
        temp_pow4_k = 1 
        temp_k = 0
        
        # Loop to find the correct k.
        # The loop continues as long as 4^(k+1) could potentially be <= N.
        # We check `temp_pow4_k <= N // 4`, which is equivalent to `4 * temp_pow4_k <= N`
        # This check uses integer arithmetic, safe for large N.
        while temp_pow4_k <= N // 4: 
             # If 4 * current power of 4 is still <= N, increment k and update power of 4.
             temp_pow4_k *= 4
             temp_k += 1
        
        # After the loop, temp_k is the largest k such that 4^k <= N.
        k = temp_k
        # temp_pow4_k now holds the value of 4^k.
        pow4_k = temp_pow4_k

        # Calculate F(4^k - 1), the sum of steps for numbers with less than k+1 digits.
        # This uses the helper function which employs memoization for efficiency.
        val_F_4k_minus_1 = self.compute_F_pow4_minus_1(k)
        
        # Calculate the contribution of numbers from 4^k up to N.
        # All these numbers x have steps(x) = k+1.
        # The count of such numbers is (N - 4^k + 1).
        count_N_k_plus_1_digits = N - pow4_k + 1
        # Total steps contributed by these numbers.
        term2 = count_N_k_plus_1_digits * (k + 1)
        
        # The total sum F(N) is F(4^k - 1) plus the contribution from numbers in [4^k, N].
        return val_F_4k_minus_1 + term2
    
    def minOperations(self, queries: List[List[int]]) -> int:
        """
        Calculates the total minimum number of operations required across all given queries.
        For each query [l, r], the array nums consists of integers from l to r inclusive.
        The minimum number of operations to reduce all elements to zero is ceil(S / 2), 
        where S = sum_{x=l}^{r} steps(x).
        S is computed efficiently using prefix sums: S = F(r) - F(l-1).
        ceil(S / 2) is computed as (S + 1) // 2 using integer division for correctness.
        """
        total_ops = 0
        
        # Iterate through each query [l, r] provided in the input list.
        for l, r in queries:
            # Calculate S_interval = sum_{x=l}^{r} steps(x).
            # This uses the compute_F function which calculates the prefix sum F(N).
            # The sum over the interval [l, r] is F(r) - F(l-1).
            S_interval = self.compute_F(r) - self.compute_F(l-1)
            
            # Calculate the minimum number of operations for the current query [l, r].
            # This is ceil(S_interval / 2).
            ops = (S_interval + 1) // 2
            
            # Add the result for the current query to the total sum.
            total_ops += ops
            
        # Return the total sum of minimum operations across all queries.
        return total_ops