from math import gcd
from typing import List

def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)

class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        k = len(target)
        full_mask = (1 << k) - 1
        
        # Precompute lcm for every non-empty mask for target array indices.
        # There are at most (2^k - 1) possible masks.
        lcm_for_mask = {}
        for mask in range(1, full_mask + 1):
            curr_lcm = 1
            for j in range(k):
                if mask & (1 << j):
                    curr_lcm = lcm(curr_lcm, target[j])
            lcm_for_mask[mask] = curr_lcm
        
        # dp[mask] : minimal cost to cover targets as represented in mask.
        dp = [float("inf")] * (full_mask + 1)
        dp[0] = 0
        
        # Process each element of nums.
        # Each element v offers candidate ways of covering some subset of targets.
        for v in nums:
            # For each nonempty subset (mask) determine cost for this element v.
            candidate_moves = []
            for mask, L in lcm_for_mask.items():
                # find smallest multiple of L which is >= v.
                # That equals: ( (v + L - 1) // L ) * L.
                new_val = ((v + L - 1) // L) * L
                cost = new_val - v
                candidate_moves.append((mask, cost))
            
            # using this element v, update the dp.
            # Note: We cannot reuse the same element more than once so we update from the previous dp.
            new_dp = dp[:]  # make a copy of current dp
            for cand_mask, cand_cost in candidate_moves:
                for state in range(full_mask + 1):
                    new_mask = state | cand_mask
                    if dp[state] + cand_cost < new_dp[new_mask]:
                        new_dp[new_mask] = dp[state] + cand_cost
            dp = new_dp
        
        return dp[full_mask]