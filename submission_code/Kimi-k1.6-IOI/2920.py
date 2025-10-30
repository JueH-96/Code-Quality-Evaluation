from collections import defaultdict
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Dictionary to store the indices of each number
        pos = defaultdict(list)
        for idx, num in enumerate(nums):
            pos[num].append(idx)
        
        min_time = float('inf')
        
        for num in pos:
            indices = pos[num]
            k = len(indices)
            if k == 1:
                # Only one occurrence, the gap is the entire array length minus 1
                current_max = n - 1
            else:
                gaps = []
                for i in range(k):
                    if i == k - 1:
                        # Circular gap between last and first index
                        gap = (indices[0] + n - indices[i] - 1)
                    else:
                        gap = indices[i + 1] - indices[i] - 1
                    gaps.append(gap)
                current_max = max(gaps)
            
            # Calculate the time needed for this number
            time = (current_max + 1) // 2
            if time < min_time:
                min_time = time
        
        return min_time