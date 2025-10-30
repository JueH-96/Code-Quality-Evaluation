from typing import List
import math

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_val = max(nums)
        min_val = min(nums)
        sum_nums = sum(nums)

        # If array is already equal
        if max_val == min_val:
            return 0

        # Calculate cost for a given total increments S and max increment D,
        # assuming 2 * cost1 > cost2 (Operation 2 is cheaper per pair)
        def calculate_cost_prefer_c2(S, D, c1, c2):
            # This function calculates the minimum cost to apply S total increments
            # with maximum D needed for any single element, using operations with costs c1 and c2.
            # This is optimal when 2 * c1 > c2.
            
            if S < 0 or D < 0: 
                 # Should not happen with non-negative diffs derived from T >= max_val
                 # but useful for robustness if called with arbitrary S, D.
                 return float('inf') 

            # If S >= 2 * D, we can pair up almost all increments. Max type 2 operations (k2) is floor(S/2).
            # The remaining S % 2 increments are done by type 1.
            if S >= 2 * D:
                k2 = S // 2
                k1 = S % 2
                cost = k2 * c2 + k1 * c1
            else: # S < 2 * D
                # The element needing D increments is the bottleneck.
                # We can pair each of the S-D increments needed by other elements
                # with an increment for the max-difference element. This gives S-D pairs.
                # Max k2 = S - D.
                # The max-difference element still needs D - (S - D) = 2D - S increments.
                # These remaining increments must be done by type 1.
                k2 = S - D
                k1 = 2 * D - S
                cost = k2 * c2 + k1 * c1
            
            return cost

        # --- Main logic ---
        
        # Calculate state for target = max_val (k=0)
        # Total increments needed to reach max_val
        S_m = n * max_val - sum_nums
        # Max increments needed for one element (min_val needs max_val - min_val increments)
        D_m = max_val - min_val

        # Case 1: 2 * cost1 <= cost2. Cheaper or equal to use operation 1 always.
        # Optimal target is max_val. Total increments S_m.
        if 2 * cost1 <= cost2:
            min_cost = S_m * cost1
            return min_cost % MOD

        # Case 2: 2 * cost1 > cost2. Operation 2 is cheaper per pair.
        # We need to consider different target values T >= max_val.
        # Let T = max_val + k, where k >= 0.
        # S(k) = sum(max_val + k - nums[i]) = sum(max_val - nums[i]) + nk = S_m + nk
        # D(k) = (max_val + k) - min_val = (max_val - min_val) + k = D_m + k
        # Cost C(k) = calculate_cost_prefer_c2(S_m + nk, D_m + k, cost1, cost2).

        # Candidate target 1: T = max_val (k=0)
        min_cost = calculate_cost_prefer_c2(S_m, D_m, cost1, cost2)

        # The cost function C(k) changes behavior when S(k) transitions relative to 2*D(k).
        # S(k) >= 2*D(k) <=> S_m + nk >= 2(D_m + k) <=> S_m - 2D_m + (n-2)k >= 0.
        # Let Req = 2*D_m - S_m = 2*(max_val - min_val) - (n*max_val - sum_nums).
        # The condition is (n-2)k >= Req.

        if n > 2:
            Req = 2 * D_m - S_m
            
            # If Req <= 0, (n-2)k >= Req is true for all k >= 0 (since n-2 > 0).
            # S(k) >= 2*D(k) for all k >= 0. The cost function C(k) is based on the S >= 2D branch
            # for all k >= 0. This cost increases with k, so minimum is at smallest k=0 (T=max_val).
            # This case is covered by the initial calculation.

            # If Req > 0, we need k >= Req / (n-2).
            # The smallest integer k satisfying this is k_boundary = ceil(Req / (n-2)).
            # The transition from S < 2D to S >= 2D happens at T = max_val + k_boundary.
            # The optimal target is among max_val (k=0), max_val + k_boundary (k=k_boundary),
            # and potentially max_val + k_boundary - 1 (k=k_boundary-1), if k_boundary >= 1.

            if Req > 0:
                k_boundary_raw = Req / (n - 2.0)
                k_boundary = math.ceil(k_boundary_raw)
                
                # Candidate T = max_val + k_boundary
                # This corresponds to k = k_boundary
                S_tb = S_m + n * k_boundary
                D_tb = D_m + k_boundary
                cost_tb = calculate_cost_prefer_c2(S_tb, D_tb, cost1, cost2)
                min_cost = min(min_cost, cost_tb)

                # Candidate T = max_val + k_boundary - 1
                # This corresponds to k = k_boundary - 1.
                # Only check if k_boundary - 1 is a valid non-negative value for k.
                if k_boundary >= 1:
                    k_tbm1 = k_boundary - 1
                    S_tbm1 = S_m + n * k_tbm1
                    D_tbm1 = D_m + k_tbm1
                    cost_tbm1 = calculate_cost_prefer_c2(S_tbm1, D_tbm1, cost1, cost2)
                    min_cost = min(min_cost, cost_tbm1)

        return min_cost % MOD