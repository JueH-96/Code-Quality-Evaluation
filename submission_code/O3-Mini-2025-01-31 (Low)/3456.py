from typing import List
import functools

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        @functools.lru_cache(maxsize=None)
        def dfs(i: int, transitions_used: int, last: int) -> int:
            # last == -1 will denote that we haven't picked any element yet.
            if i == n:
                return 0
            
            # option 1: skip current element
            best = dfs(i+1, transitions_used, last)
            
            # option 2: take current element if allowed:
            cur = nums[i]
            # Determine if taking cur causes a transition.
            # If last == -1 then no transition is counted.
            if last == -1 or cur == last:
                # No transition cost.
                best = max(best, 1 + dfs(i+1, transitions_used, cur))
            else:
                # only take if adding a transition is within bound.
                if transitions_used < k:
                    best = max(best, 1 + dfs(i+1, transitions_used+1, cur))
            return best
        
        return dfs(0, 0, -1)