from collections import defaultdict
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        counts = defaultdict(int)
        for num in nums:
            counts[num] += 1
        
        ans = 0
        visited = set()

        for num in counts:
            if num in visited:
                continue
            
            curr = num
            path_len = 0
            path = []
            
            while curr in counts:
                path.append(curr)
                visited.add(curr)
                if counts[curr] == 0:
                    break
                counts[curr] -= 1
                if curr == 1:
                    break
                if curr > 1000000000:
                    break
                
                next_val = curr * curr
                if next_val not in counts:
                    break
                curr = next_val
            
            path_len = len(path)
            
            if path_len == 1:
                ans = max(ans, 1)
            elif path_len % 2 == 1:
                ans = max(ans, path_len)
            else:
                ans = max(ans, path_len - 1)
        
        return ans