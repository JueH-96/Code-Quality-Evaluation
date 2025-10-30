from typing import List
from collections import defaultdict, Counter
import math

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Compute total counts of each number
        total_counts = Counter(nums)
        
        # Compute prefix counts
        prefix_counts = [defaultdict(int)]
        current = defaultdict(int)
        for num in nums:
            current = current.copy()
            current[num] += 1
            prefix_counts.append(current)
        
        result = 0
        
        for i in range(n):
            v = nums[i]
            
            # Counts to the left of i
            left_v = prefix_counts[i][v]
            left_total = i
            left_non_v = left_total - left_v
            
            # Counts to the right of i
            right_v = total_counts[v] - prefix_counts[i+1][v]
            right_total = n - i - 1
            right_non_v = right_total - right_v
            
            # Iterate over possible a and b
            for a in range(0, min(2, left_v) + 1):
                for b in range(0, min(2, right_v) + 1):
                    if a + b >= 2:
                        # Calculate combinations
                        comb_left_v = math.comb(left_v, a)
                        comb_left_non_v = math.comb(left_non_v, 2 - a) if (2 - a) <= left_non_v else 0
                        comb_right_v = math.comb(right_v, b)
                        comb_right_non_v = math.comb(right_non_v, 2 - b) if (2 - b) <= right_non_v else 0
                        
                        # If combinations are valid, add to result
                        if comb_left_v and comb_left_non_v and comb_right_v and comb_right_non_v:
                            count = comb_left_v * comb_left_non_v * comb_right_v * comb_right_non_v
                            result = (result + count) % MOD
        return result