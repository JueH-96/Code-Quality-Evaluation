from typing import List

class Solution:
  def minOperations(self, queries: List[List[int]]) -> int:
    """
    Calculates the total minimum operations for a list of queries.

    The method relies on calculating the total number of "division-by-4" steps
    required for all numbers in a given range [l, r]. The minimum number of
    operations is then derived from this total.

    To efficiently calculate the sum of these steps for large ranges, a recursive
    approach with memoization is used for a prefix sum function T(n), where
    T(n) = sum of steps for numbers from 1 to n.
    """
    
    memo_f = {0: 0}
    memo_T = {0: 0}

    def f(n: int) -> int:
        """Calculates the number of divisions by 4 to make n zero."""
        if n in memo_f:
            return memo_f[n]
        res = 1 + f(n // 4)
        memo_f[n] = res
        return res

    def T(n: int) -> int:
        """Calculates the sum of f(i) for i from 1 to n."""
        if n <= 0:
            return 0
        if n in memo_T:
            return memo_T[n]
        
        m = n // 4
        rem = n % 4
        
        # T(n) is derived from the recurrence relation:
        # T(n) = n + 4*T(n//4) - (3 - n%4)*f(n//4)
        res = n + 4 * T(m) - (3 - rem) * f(m)
        memo_T[n] = res
        return res
    
    total_result = 0
    for l, r in queries:
        # Calculate S, the total "work" for the range [l, r]
        s = T(r) - T(l - 1)
        
        # The minimum operations is ceil(S/2)
        query_result = (s + 1) // 2
        total_result += query_result
        
    return total_result