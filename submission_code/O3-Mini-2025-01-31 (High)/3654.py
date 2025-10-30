from typing import List
import math

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        # The idea is to decide, for every element nums[i], whether to:
        #   0: do nothing, or
        #   1: apply Operation 1 only (divide by 2, rounding up),
        #   2: apply Operation 2 only (subtract k, if possible),
        #   3: apply both operations on that element (in the best possible order).
        #
        # Each operation can be applied at most once per index.
        # We have a total budget of op1 uses for Operation 1 and op2 uses for Operation 2.
        #
        # Instead of directly minimizing the final sum, note that the overall sum is fixed,
        # and each operation “saves” (reduces) a certain amount from the original value.
        # If we let "reduction" be the amount subtracted from the original number by applying an op,
        # then the final sum = total_sum - (total reduction).
        #
        # For any element v, the baseline (do nothing) gives final value v (reduction 0).
        # Operation 1 ("op1 only") gives final value = ceil(v/2) i.e. (v+1)//2 
        #    => reduction = v - ceil(v/2).
        # Operation 2 ("op2 only") is allowed only if v >= k, and yields final value = v - k
        #    => reduction = k.
        # For applying both ("both") on an element v (allowed if v >= k), the order matters.
        # We have two orders:
        #   Order A: op1 then op2:
        #         new_v = ceil(v/2)
        #         then if ceil(v/2) >= k, op2 gives new_v - k.
        #         Final value = ceil(v/2) - k, so reduction = v - ceil(v/2) + k.
        #   Order B: op2 then op1:
        #         first, op2: v -> v - k   (allowed if v >= k)
        #         then op1: new_v = ceil((v-k)/2)
        #         Final value = ceil((v-k)/2), so reduction = v - ceil((v-k)/2).
        #
        # We always want the order that minimizes the final value (i.e. maximizes reduction).
        # Thus, if v >= k:
        #    - op1 only gives reduction = v - ceil(v/2)
        #    - op2 only gives reduction = k
        #    - both (if allowed) gives reduction = 
        #          max(  [if ceil(v/2) >= k then v - ceil(v/2) + k] ,  v - ceil((v-k)/2) )
        # For v < k, op2 or any op involving op2 are not allowed.
        #
        # Then we use a DP (multi-choice knapSack) where for each number we choose one option.
        # The DP state is dp[i][a][b] = maximum total reduction after processing i items
        # while having used a times Operation 1 and b times Operation 2.
        # Since n and operations budgets are small (<= 100), we can use a 2D DP iterated over the array.
        
        total = sum(nums)
        # We'll use dp[a][b] = maximum total reduction achieved using 'a' op1 uses and 'b' op2 uses
        # after processing some of the numbers.
        dp = [[float('-inf')] * (op2 + 1) for _ in range(op1 + 1)]
        dp[0][0] = 0
        
        # Process each number one-by-one.
        for v in nums:
            newdp = [[float('-inf')] * (op2 + 1) for _ in range(op1 + 1)]
            for a in range(op1 + 1):
                for b in range(op2 + 1):
                    if dp[a][b] == float('-inf'):
                        continue
                    current = dp[a][b]
                    # Option 0: do nothing.
                    if newdp[a][b] < current:
                        newdp[a][b] = current
                    # Option 1: Operation 1 only.
                    if a < op1:
                        # Compute new value when doing op1 only: new_v = ceil(v/2)
                        # Reduction = v - new_v.
                        reduction = v - ((v + 1) // 2)
                        if newdp[a + 1][b] < current + reduction:
                            newdp[a + 1][b] = current + reduction
                    # For operations that involve op2, we require v >= k.
                    if v >= k:
                        # Option 2: Operation 2 only.
                        if b < op2:
                            # Final value = v - k, so reduction = k.
                            reduction = k
                            if newdp[a][b + 1] < current + reduction:
                                newdp[a][b + 1] = current + reduction
                        # Option 3: Apply both operations on the same index.
                        if a < op1 and b < op2:
                            # Two orders exist. We'll compute the reduction for each and choose the maximum.
                            # Order A: op1 then op2.
                            # Allowed only if first op (v->ceil(v/2)) still allows op2: i.e. ceil(v/2) >= k.
                            candidate1 = None
                            if ((v + 1) // 2) >= k:
                                candidate1 = v - ((v + 1) // 2) + k
                            # Order B: op2 then op1 is always allowed when v >= k.
                            candidate2 = v - (((v - k) + 1) // 2)
                            if candidate1 is None:
                                both_reduction = candidate2
                            else:
                                both_reduction = max(candidate1, candidate2)
                            if newdp[a + 1][b + 1] < current + both_reduction:
                                newdp[a + 1][b + 1] = current + both_reduction
            dp = newdp
        
        best_reduction = 0
        for a in range(op1 + 1):
            for b in range(op2 + 1):
                if dp[a][b] > best_reduction:
                    best_reduction = dp[a][b]
        
        # The minimal final sum is total sum minus total reduction we can achieve.
        return total - best_reduction