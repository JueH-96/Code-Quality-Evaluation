class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        from functools import lru_cache
        
        n = len(nums)
        
        @lru_cache(None)
        def dfs(pos, last_index, changes_left):
            if pos == n:
                return 0
            
            # Option 1: skip nums[pos]
            best = dfs(pos + 1, last_index, changes_left)
            
            # Option 2: pick nums[pos] if possible
            # Determine if picking causes a change
            if last_index == -1 or nums[pos] == nums[last_index]:
                cost = 0
            else:
                cost = 1
            
            if changes_left >= cost:
                best = max(
                    best,
                    1 + dfs(pos + 1, pos, changes_left - cost)
                )
            
            return best
        
        return dfs(0, -1, k)