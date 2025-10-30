class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        """
        We want an array of length n of distinct positive integers such that no two
        elements sum to target, and we want the minimum possible sum of such an array
        modulo 10^9+7.

        Key Insight:
        - From the range [1..target-1], each pair (x, target-x) cannot both be chosen.
        - We can pick at most one number from each such pair.
        - Let L = floor((target-1)/2).
          Then we can safely choose the numbers 1, 2, ..., L (none of these pair up among
          themselves to sum to target).
        - If n <= L, we can take the first n positive integers and be done.
        - If n > L, we take those L, then we must skip [L+1..target-1] (since each of those 
          would have a partner in [1..L]), and pick the remaining (n - L) numbers starting 
          from target, target+1, etc.
        
        Steps:
        1) Compute L = (target - 1) // 2.
        2) If n <= L, the sum is simply sum of 1..n = n*(n+1)//2.
        3) Otherwise, we sum 1..L plus target..(target + (n - L) - 1).
        
        We'll compute the result in O(1) time and take everything modulo 1e9+7.
        """
        mod = 10**9 + 7
        L = (target - 1) // 2

        def triangular_sum(x):
            # Returns x*(x+1)//2 mod
            return (x * (x + 1) // 2) % mod

        if n <= L:
            # Sum of first n natural numbers
            return triangular_sum(n) % mod
        else:
            # Sum of first L natural numbers
            part1 = triangular_sum(L)
            # We add the next (n-L) numbers starting from 'target'
            # That sum is (n-L)*target + sum of 0..(n-L-1)
            remaining = n - L
            part2 = (remaining * target) % mod
            # sum of 0..(remaining - 1) = (remaining-1)*remaining//2
            part2 += (remaining * (remaining - 1) // 2) % mod
            return (part1 + part2) % mod