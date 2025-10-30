from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # dp maps (alt_sum, next_parity) -> (max_product, has_non_empty)
        # next_parity: 0 means the next element will be placed at even index in subsequence
        dp = {(0, 0): (1, False)}  # start with empty subsequence
        
        for num in nums:
            # Work on a snapshot to avoid overwriting during iteration
            snapshot = list(dp.items())
            for (cur_sum, par), (cur_prod, cur_non_empty) in snapshot:
                # Attempt to take this num
                # If next_parity == 0, we add num; if 1, we subtract num
                new_sum = cur_sum + (num if par == 0 else -num)
                new_prod = cur_prod * num
                if new_prod <= limit:
                    new_par = 1 - par
                    key = (new_sum, new_par)
                    # Taking makes it non-empty
                    new_non_empty = True
                    if key in dp:
                        old_prod, old_non_empty = dp[key]
                        # Prefer larger product; if tie, non-empty is better
                        if new_prod > old_prod or (new_prod == old_prod and new_non_empty and not old_non_empty):
                            dp[key] = (new_prod, new_non_empty)
                    else:
                        dp[key] = (new_prod, new_non_empty)
            # Also consider starting a new subsequence at this num alone
            # Actually the above loop from empty state handles that already
        
        # Extract the best product for alt sum == k, from either parity,
        # but ensure it's non-empty
        ans = -1
        for par in (0, 1):
            key = (k, par)
            if key in dp:
                prod, non_empty = dp[key]
                if non_empty and prod > ans:
                    ans = prod
        
        return ans