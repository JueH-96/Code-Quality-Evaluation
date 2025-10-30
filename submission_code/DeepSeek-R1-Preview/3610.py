from typing import List
from collections import defaultdict

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            window = nums[i:i+k]
            # Count frequency
            freq = defaultdict(int)
            for num in window:
                freq[num] += 1
            # Create list of (frequency, value) tuples
            lst = [ (f, num) for num, f in freq.items() ]
            # Sort by frequency descending, then value descending
            lst.sort(reverse=True)
            # Take top x elements
            take = min(x, len(lst))
            top_x = lst[:take]
            # Calculate the sum
            total = sum(f * num for f, num in top_x)
            result.append(total)
        return result