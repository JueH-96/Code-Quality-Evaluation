from typing import List

class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        min_positive = float('inf')
        
        for i in range(n):
            start_j = i + l - 1
            if start_j >= n:
                break
            end_j = min(i + r - 1, n - 1)
            for j in range(start_j, end_j + 1):
                s = prefix[j + 1] - prefix[i]
                if s == 1:
                    return 1
                if s > 0 and s < min_positive:
                    min_positive = s
        
        return min_positive if min_positive != float('inf') else -1