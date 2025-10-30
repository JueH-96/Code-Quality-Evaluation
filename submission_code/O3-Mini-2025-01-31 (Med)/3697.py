from math import gcd
from typing import List

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        # Number of targets (m <= 4)
        m = len(target)
        full_mask = (1 << m) - 1

        # Helper function: compute lcm of two numbers
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        # Precompute LCM for every nonempty subset S ⊆ target 
        # We'll represent S as a bitmask, with bits 0..m-1.
        lcm_table = [0] * (1 << m)  # index 0 is unused.
        for mask in range(1, 1 << m):
            current_lcm = 1
            bit = 0
            temp = mask
            while temp:
                if temp & 1:
                    current_lcm = lcm(current_lcm, target[bit])
                temp //= 2
                bit += 1
            lcm_table[mask] = current_lcm
        
        # Precompute union_table: union_table[s][c] = s|c for s in 0..full_mask and c in 0..full_mask.
        size = 1 << m
        union_table = [[0]*size for _ in range(size)]
        for s in range(size):
            for c in range(size):
                union_table[s][c] = s | c

        # Initialize dp: dp[mask] = minimal cost to cover the targets in 'mask'
        INF = 10**9  # given constraints, operations cost will not exceed this value.
        dp = [INF] * size
        dp[0] = 0  # no cost to cover an empty set.

        # Process each element in nums (each can be used at most once).
        # For the current element v, we compute candidate possibilities:
        # for every nonempty bitmask c (which corresponds to a subset of target indices),
        # compute the minimal extra cost to convert v into an integer that is a multiple of LCM(targets in subset)
        # and then update the dp.
        for v in nums:
            # Create candidate possibilities for this element.
            # Each candidate possibility is a tuple (cand_mask, cost)
            cand_list = []
            # Loop over all nonzero bitmasks (nonempty subset of target indices)
            for c in range(1, size):
                L = lcm_table[c]
                # Compute the smallest multiple of L that's >= v.
                # That is, cost = L * ceil(v / L) - v.
                # (Using integer arithmetic: ceil(v/L) = (v+L-1)//L)
                cost = ((v + L - 1) // L) * L - v
                cand_list.append((c, cost))
            
            # For the knapSack/update, we need to try either not using v or using it with one of its candidate possibilities.
            new_dp = dp[:]  # copy current dp (i.e. not using v)
            # For each candidate possibility from this element:
            for cand_mask, cost in cand_list:
                base_cost = cost  # cost offset if we use v with candidate possibility cand_mask.
                # Try combining with every previous state.
                for prev_mask in range(size):
                    # Only try if dp[prev_mask] is already valid.
                    if dp[prev_mask] == INF:
                        continue
                    new_mask = union_table[prev_mask][cand_mask]
                    new_val = dp[prev_mask] + base_cost
                    if new_val < new_dp[new_mask]:
                        new_dp[new_mask] = new_val
            dp = new_dp
        
        return dp[full_mask]