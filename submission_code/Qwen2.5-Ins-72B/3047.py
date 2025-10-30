from typing import List
import math
from collections import defaultdict

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(x):
            return int(math.isqrt(x)) ** 2 == x
        
        n = len(nums)
        max_sum = max(nums)
        
        # Dictionary to store subsets that form complete sets
        subsets = defaultdict(list)
        
        for i in range(n):
            for j in range(i + 1, n):
                if is_perfect_square(nums[i] * nums[j]):
                    subsets[(i, j)].append(i)
                    subsets[(i, j)].append(j)
        
        # Check for larger complete subsets
        for subset in subsets.values():
            subset = list(set(subset))
            for i in subset:
                for j in subset:
                    if i != j and not is_perfect_square(nums[i] * nums[j]):
                        break
                else:
                    continue
                break
            else:
                max_sum = max(max_sum, sum(nums[i] for i in subset))
        
        return max_sum