from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        count = Counter(nums)
        visited = set()
        max_length = 1
        
        for num in nums:
            if num in visited:
                continue
            
            # Special case for 1
            if num == 1:
                max_length = max(max_length, count[num] - (1 - count[num] % 2))
                visited.add(num)
                continue
            
            current_length = 1
            x = num
            visited.add(x)
            
            while x * x in count:
                visited.add(x * x)
                if count[x * x] < 2:
                    break
                current_length += 2
                x *= x
            
            max_length = max(max_length, current_length)
        
        return max_length