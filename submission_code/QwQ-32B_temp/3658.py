from typing import List

class Solution:
    def minDifference(self, nums: List[int]) -> int:
        non_missing = []
        for idx, num in enumerate(nums):
            if num != -1:
                non_missing.append((idx, num))
        
        n = len(non_missing)
        if n <= 1:
            return 0
        
        max_d = 0
        for i in range(n - 1):
            pos_i, a = non_missing[i]
            pos_j, b = non_missing[i + 1]
            steps = pos_j - pos_i
            distance = abs(b - a)
            required = (distance + steps - 1) // steps
            if required > max_d:
                max_d = required
        
        return max_d