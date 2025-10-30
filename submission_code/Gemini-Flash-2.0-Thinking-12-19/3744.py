from typing import List

class Solution:
    @staticmethod
    def ops(x: int) -> int:
        """
        Calculates the number of operations needed to reduce x to 0 if paired optimally.
        This is the smallest integer k such that floor(x / 4^k) == 0.
        Equivalent to floor(log4(x)) + 1 for x > 0.
        Since max x is 10^9, max ops(x) is ops(10^9) = 15.
        """
        if x == 0:
            return 0
        count = 0
        temp_x = x
        while temp_x > 0:
            temp_x //= 4
            count += 1
        return count

    @staticmethod
    def SumPrefix(K: int) -> int:
        """
        Calculates the sum of ops(x) for x from 1 up to 4^K - 1.
        SumPrefix(K) = sum_{x=1}^{4^K - 1} ops(x) for K >= 0.
        This is the sum of ops for all numbers x < 4^K.
        ops(x) = k+1 for x in [4^k, 4^{k+1} - 1]. There are 3 * 4^k such numbers.
        SumPrefix(K) = sum_{k=0}^{K-1} (k+1) * (3 * 4^k)
        Using the formula: ((3*K - 1) * 4^K + 1) // 3 for K >= 0.
        """
        # Sum from 1 to 4^K - 1. If K <= 0, this range is empty or [1, 0], sum is 0.
        if K <= 0:
            return 0
        
        # K >= 1
        # We need 4^K. Maximum K is M-1 = ops(N)-1 <= 15-1 = 14 (for N=10^9). pow(4, 14) is fine.
        power_of_4_K = pow(4, K)
        
        # The formula ((3 * K - 1) * power_of_4_K + 1) is guaranteed to be divisible by 3.
        # (3K - 1) * 4^K + 1 mod 3 = (0 - 1) * 1^K + 1 mod 3 = -1 + 1 mod 3 = 0.
        return ((3 * K - 1) * power_of_4_K + 1) // 3

    @staticmethod
    def F(N: int) -> int:
        """
        Calculates the sum of ops(x) for x from 1 up to N.
        F(N) = sum_{x=1}^N ops(x) for N >= 0.
        F(0) = 0.
        For N >= 1, let M = ops(N). This means 4^(M-1) <= N < 4^M.
        The numbers from 1 to N can be split into two groups:
        1. Numbers x such that ops(x) < M. These are numbers in the range [1, 4^(M-1) - 1].
           The sum of ops for this group is SumPrefix(M-1).
        2. Numbers x such that ops(x) = M. These are numbers in the range [4^(M-1), N].
           The sum of ops for this group is M * (N - 4^(M-1) + 1).
        F(N) = SumPrefix(M-1) + M * (N - 4^(M-1) + 1) for N >= 1.
        """
        if N == 0:
            return 0
        
        M = Solution.ops(N) # M >= 1 since N >= 1
        
        # Sum of ops(x) for x in [1, 4^(M-1) - 1] is SumPrefix(M-1)
        # M-1 >= 0 since M >= 1
        sum_prefix_m_minus_1 = Solution.SumPrefix(M - 1)
        
        # Sum of ops(x) for x in [4^(M-1), N]
        # All numbers in this range have ops(x) = M.
        # The lower bound 4^(M-1) is pow(4, M-1). M-1 >= 0, so pow(4, M-1) >= 1.
        power_of_4_M_minus_1 = pow(4, M - 1) 
        
        # Number of terms is N - 4^(M-1) + 1
        sum_current_level = M * (N - power_of_4_M_minus_1 + 1)
        
        total_sum = sum_prefix_m_minus_1 + sum_current_level
        
        return total_sum

    def minOperations(self, queries: List[List[int]]) -> int:
        total_sum_of_min_ops = 0
        
        for l, r in queries:
            # The initial array for the query is [l, l+1, ..., r].
            # The total number of 'ops steps' required across all elements is sum(ops(x) for x in range(l, r + 1)).
            # This sum can be calculated efficiently using the prefix sum function F:
            # sum(ops(x) for x in range(l, r + 1)) = sum(ops(x) for x in range(1, r + 1)) - sum(ops(x) for x in range(1, l))
            # = F(r) - F(l - 1).
            sum_l_r = Solution.F(r) - Solution.F(l - 1)
            
            # The minimum number of operations required for a given array is half the total sum of ops(x) values.
            # This is because each operation reduces the total 'ops steps' by 2 (one for each element in the pair).
            # The sum_l_r is guaranteed to be even (as argued in thought process), so integer division // 2 is correct.
            min_ops_for_query = sum_l_r // 2
            
            total_sum_of_min_ops += min_ops_for_query
            
        return total_sum_of_min_ops