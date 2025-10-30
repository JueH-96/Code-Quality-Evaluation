import bisect
from collections import defaultdict
from typing import List

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Create a dictionary to group indices by each number's occurrences
        index_map = defaultdict(list)
        for idx, num in enumerate(nums):
            index_map[num].append(idx)
        
        max_length = 0
        
        # Process each number's list of indices
        for num, indices in index_map.items():
            m = len(indices)
            # Calculate the adjusted indices list b
            b = [indices[j] - j for j in range(m)]
            
            for j in range(m):
                target = b[j] + k
                # Find the rightmost position where b[l] <= target using binary search
                ip = bisect.bisect_right(b, target, j)
                l = ip - 1
                if l >= j:
                    current_length = l - j + 1
                    if current_length > max_length:
                        max_length = current_length
        
        return max_length