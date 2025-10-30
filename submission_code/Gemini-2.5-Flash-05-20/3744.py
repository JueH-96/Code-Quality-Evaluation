import math
from typing import List

class Solution:
    # Precompute powers of 4 for efficiency.
    # Max N is 10^9. The maximum depth K_max for N <= 10^9 is 15 (since 4^14 < 10^9 < 4^15).
    # We need powers of 4 up to 4^(K_max-1), which is 4^14.
    # Our list covers 4^0 to 4^15, so it's sufficient.
    POWERS_OF_4 = [4**i for i in range(16)] 

    def _get_depth(self, n: int) -> int:
        """
        Calculates the 'depth' of a number n, which is the minimum operations
        to reduce n to 0 if it's always paired.
        depth(n) = k means 4^(k-1) <= n < 4^k.
        This is equivalent to floor(log_4(n)) + 1 for n >= 1.
        Using integer arithmetic for robustness and speed.
        """
        if n == 0:
            return 0
        
        # floor(log_4(n)) = floor(log_2(n) / 2)
        # For n >= 1, n.bit_length() - 1 gives floor(log_2(n)).
        # So, floor(log_4(n)) = (n.bit_length() - 1) // 2
        # The depth is floor(log_4(n)) + 1.
        return (n.bit_length() - 1) // 2 + 1

    def _get_S(self, N: int) -> int:
        """
        Calculates S(N) = sum_{x=1 to N} depth(x).
        """
        if N == 0:
            return 0
        
        total_sum_depth = 0
        
        # K_max is the depth of N itself.
        K_max = self._get_depth(N)
        
        # Add contribution from numbers whose depth ranges from 1 to K_max-1.
        # For a fixed depth k, numbers are in the range [4^(k-1), 4^k - 1].
        # The count of such numbers is (4^k - 1) - 4^(k-1) + 1 = 3 * 4^(k-1).
        for k in range(1, K_max):
            count_for_this_depth = 3 * self.POWERS_OF_4[k-1]
            total_sum_depth += k * count_for_this_depth
            
        # Add contribution from numbers with the maximum depth K_max within [1, N].
        # These are numbers in the range [4^(K_max-1), N].
        # The count is N - 4^(K_max-1) + 1.
        count_in_last_range = N - self.POWERS_OF_4[K_max-1] + 1
        total_sum_depth += K_max * count_in_last_range
        
        return total_sum_depth

    def minOperations(self, queries: List[List[int]]) -> int:
        """
        Determines the minimum number of operations required to reduce all elements
        of the array to zero for each query [l, r] and returns the sum of these results.
        """
        total_min_operations_across_all_queries = 0
        
        for l, r in queries:
            # Calculate the sum of depths for numbers from 1 to r.
            sum_depth_up_to_r = self._get_S(r)
            
            # Calculate the sum of depths for numbers from 1 to l-1.
            sum_depth_up_to_l_minus_1 = self._get_S(l - 1)
            
            # The sum of depths for numbers in the range [l, r] is S(r) - S(l-1).
            current_query_total_depth = sum_depth_up_to_r - sum_depth_up_to_l_minus_1
            
            # The minimum number of operations is ceil(total_depth / 2).
            # For positive integers A, B, ceil(A/B) can be calculated as (A + B - 1) // B.
            # Here, B=2, so it's (current_query_total_depth + 1) // 2.
            current_query_min_ops = (current_query_total_depth + 1) // 2
            
            total_min_operations_across_all_queries += current_query_min_ops
            
        return total_min_operations_across_all_queries