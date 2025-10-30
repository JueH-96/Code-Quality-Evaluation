import math

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        pos = []
        for i, num_val in enumerate(nums):
            if num_val == 1:
                pos.append(i)
        
        m = len(pos)
        ans = float('inf')

        if k == 0:
            return 0
        
        # Scenario 0: k_orig = 0. All k items from changes.
        # Alice picks an arbitrary aliceIndex (where nums[aliceIndex] is 0).
        # Each of k items costs 2 moves. Total 2*k.
        # Requires maxChanges >= k.
        if maxChanges >= k:
            ans = min(ans, 2 * k)
        
        # Scenario 1: k_orig = 1. k-1 items from changes, 1 original item.
        # The original item is at aliceIndex, so it's free.
        # Cost for k-1 changed items is 2*(k-1).
        # Requires maxChanges >= k-1 and m >= 1.
        if m >= 1: # Check if at least one '1' exists
            # If k=1, this is 2*(1-1)=0. Needs maxChanges >= 0.
            # If k>1, this is 2*(k-1). Needs maxChanges >= k-1.
            if maxChanges >= (k - 1):
                 ans = min(ans, 2 * (k - 1))
        
        # Scenario 2: k_orig = 2. k-2 items from changes, 2 original items.
        # One original item is at aliceIndex (free). Second original item is moved.
        # Cost for k-2 changed items is 2*(k-2).
        # Cost for moving second original item is min(abs(p_i - p_j)). This is min(pos[i+1]-pos[i]).
        # Requires maxChanges >= k-2 and m >= 2.
        if k >= 2 and m >= 2: # Check if k original items possible and k-2 is non-negative
            if maxChanges >= (k - 2):
                min_adj_diff = float('inf')
                for i in range(m - 1):
                    min_adj_diff = min(min_adj_diff, pos[i+1] - pos[i])
                ans = min(ans, 2 * (k - 2) + min_adj_diff)

        # Scenario 3: k_orig = 3. k-3 items from changes, 3 original items.
        # One original item (median of three) at aliceIndex (free). Two others moved.
        # Cost for k-3 changed items is 2*(k-3).
        # Cost for moving two original items is min( (p_median-p_left) + (p_right-p_median) ) = min(p_right-p_left).
        # This is min(pos[i+2]-pos[i]).
        # Requires maxChanges >= k-3 and m >= 3.
        if k >= 3 and m >= 3: # Check if k original items possible and k-3 is non-negative
            if maxChanges >= (k - 3):
                min_skip1_diff = float('inf')
                for i in range(m - 2):
                    min_skip1_diff = min(min_skip1_diff, pos[i+2] - pos[i])
                ans = min(ans, 2 * (k - 3) + min_skip1_diff)
            
        return ans