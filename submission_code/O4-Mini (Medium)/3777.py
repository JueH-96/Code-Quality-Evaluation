from typing import List
from collections import defaultdict

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # dp maps (alternating_sum, parity) -> dict: product -> nonempty_flag
        # parity = current length mod 2 (0 means next index is even, 1 means next is odd)
        dp = {(0, 0): {1: False}}  # start with empty subsequence: sum=0, parity=0, product=1, empty flag
        
        for num in nums:
            # take a snapshot of current dp entries to avoid using newly added states in this iteration
            snapshot = [ (s, p, list(pr_map.items())) 
                         for (s, p), pr_map in dp.items() ]
            
            for s, parity, pr_list in snapshot:
                for pr, _empty_flag in pr_list:
                    new_pr = pr * num
                    if new_pr > limit:
                        continue
                    # compute new alternating sum and parity
                    if parity == 0:
                        new_s = s + num
                    else:
                        new_s = s - num
                    new_parity = parity ^ 1
                    # insert into dp
                    key = (new_s, new_parity)
                    if key not in dp:
                        dp[key] = {new_pr: True}
                    else:
                        # mark non-empty subsequences with True
                        if new_pr in dp[key]:
                            dp[key][new_pr] = dp[key][new_pr] or True
                        else:
                            dp[key][new_pr] = True
        
        # check results for sum == k (parity can be 0 or 1), only non-empty flags (True)
        res = -1
        for parity in (0, 1):
            pr_map = dp.get((k, parity), {})
            for pr, non_empty in pr_map.items():
                if non_empty and pr <= limit:
                    if pr > res:
                        res = pr
        
        return res