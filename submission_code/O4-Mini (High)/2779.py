from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        ans = 0
        res = []
        
        for idx, color in queries:
            # Remove existing same-color adjacency involving idx
            if idx - 1 >= 0 and nums[idx - 1] == nums[idx] and nums[idx] != 0:
                ans -= 1
            if idx + 1 < n and nums[idx] == nums[idx + 1] and nums[idx] != 0:
                ans -= 1
            
            # Color the index
            nums[idx] = color
            
            # Add new same-color adjacency involving idx
            if idx - 1 >= 0 and nums[idx - 1] == nums[idx] and nums[idx] != 0:
                ans += 1
            if idx + 1 < n and nums[idx] == nums[idx + 1] and nums[idx] != 0:
                ans += 1
            
            res.append(ans)
        
        return res