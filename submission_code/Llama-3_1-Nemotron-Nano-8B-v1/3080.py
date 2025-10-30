from typing import List

class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute the suffix AND array
        suffix_and = [0] * n
        suffix_and[-1] = nums[-1]
        for i in range(n-2, -1, -1):
            suffix_and[i] = nums[i] & suffix_and[i+1]
        
        # Check if the entire array's AND is non-zero
        total_and = suffix_and[0]
        if total_and != 0:
            return 1
        
        count = 0
        current_start = 0
        current_and = suffix_and[0]
        
        for i in range(current_start, n):
            if i == current_start:
                current_and = nums[i]
            else:
                current_and &= nums[i]
            
            if current_and == 0:
                if i == n - 1:
                    count += 1
                    break
                else:
                    next_and = suffix_and[i + 1]
                    if next_and == 0:
                        count += 1
                        current_start = i + 1
                        current_and = next_and
                        # Reset i to current_start - 1 to reprocess from the new start
                        i = current_start - 1
        
        return count