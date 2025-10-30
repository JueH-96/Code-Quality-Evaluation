from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        nums.sort()
        n = len(nums)
        max_freq = 0
        for i in range(n):
            freq = 0
            ops_left = numOperations
            target = nums[i]
            
            
            indices = []
            for j in range(n):
                indices.append((abs(nums[j]-target),j))
            
            indices.sort()
            
            
            
            for diff, j in indices:
                if nums[j] <= target:
                    freq += 1
                else:
                    if ops_left > 0:
                        if nums[j] - target <= k:
                            freq += 1
                            ops_left -= 1
                        
            
            max_freq = max(max_freq, freq)
        return max_freq