from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        
        for i in range(n - k + 1):
            # Extract the current window
            window = nums[i:i+k]
            # Count frequencies
            freq = Counter(window)
            # Sort by descending frequency, then by descending value
            # Then pick the top x distinct elements
            top_items = sorted(freq.items(), key=lambda iv: (-iv[1], -iv[0]))[:x]
            # Sum up all occurrences of these top elements
            s = sum(val * count for val, count in top_items)
            answer.append(s)
        
        return answer