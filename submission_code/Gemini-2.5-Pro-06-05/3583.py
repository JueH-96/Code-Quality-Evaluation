import bisect
from typing import List

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Based on constraints, the maximum value in nums is 5 * 10^4.
        MAX_VAL = 50000

        # counts[i] stores the frequency of number i in the input nums.
        counts = [0] * (MAX_VAL + 1)
        for num in nums:
            counts[num] += 1

        # multiples_of[i] stores the count of numbers in nums that are multiples of i.
        # This is computed efficiently using a sieve-like method.
        multiples_of = [0] * (MAX_VAL + 1)
        for i in range(1, MAX_VAL + 1):
            for j in range(i, MAX_VAL + 1, i):
                multiples_of[i] += counts[j]

        # count_exact_gcd[g] stores the number of pairs (a, b) with gcd(a, b) == g.
        # We compute this using the principle of inclusion-exclusion, iterating g downwards.
        count_exact_gcd = [0] * (MAX_VAL + 1)
        for g in range(MAX_VAL, 0, -1):
            num_multiples = multiples_of[g]
            if num_multiples < 2:
                continue
            
            # Total pairs with GCD being a multiple of g is C(num_multiples, 2).
            num_pairs_multiple_of_g = num_multiples * (num_multiples - 1) // 2
            
            # Subtract pairs where GCD is a multiple of 2g, 3g, etc.,
            # which we have already computed in previous iterations.
            for k in range(2 * g, MAX_VAL + 1, g):
                num_pairs_multiple_of_g -= count_exact_gcd[k]
            
            count_exact_gcd[g] = num_pairs_multiple_of_g

        # cumulative_counts[g] stores the number of pairs with GCD <= g.
        # This is the prefix sum of count_exact_gcd.
        cumulative_counts = [0] * (MAX_VAL + 1)
        current_sum = 0
        for i in range(1, MAX_VAL + 1):
            current_sum += count_exact_gcd[i]
            cumulative_counts[i] = current_sum

        ans = []
        for q in queries:
            # For a query index q, we need the (q+1)-th smallest GCD.
            # This is the smallest value `g` such that the number of pairs
            # with GCD <= g is greater than q.
            # bisect_right finds the insertion point for q, which corresponds to
            # the index of the first element strictly greater than q.
            g = bisect.bisect_right(cumulative_counts, q)
            ans.append(g)
            
        return ans