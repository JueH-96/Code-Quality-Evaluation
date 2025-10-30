from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        nums = [0] * n
        ans = 0
        res = []
        
        for idx, color in queries:
            old = nums[idx]
            # Remove contributions of old color with neighbors
            if old != 0:
                if idx > 0 and nums[idx - 1] == old:
                    ans -= 1
                if idx < n - 1 and nums[idx + 1] == old:
                    ans -= 1
            # Color the cell
            nums[idx] = color
            # Add contributions of new color with neighbors
            if idx > 0 and nums[idx - 1] == color:
                ans += 1
            if idx < n - 1 and nums[idx + 1] == color:
                ans += 1
            
            res.append(ans)
        
        return res