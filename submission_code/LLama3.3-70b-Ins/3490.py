from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_length = 0
        
        # Iterate over all possible subsequences
        for mask in range(1 << n):
            subsequence = [nums[i] for i in range(n) if (mask & (1 << i))]
            
            # Check if the subsequence is valid
            if len(subsequence) > 1:
                valid = True
                parity = (subsequence[0] + subsequence[1]) % 2
                for i in range(1, len(subsequence) - 1):
                    if (subsequence[i] + subsequence[i + 1]) % 2 != parity:
                        valid = False
                        break
                if valid:
                    max_length = max(max_length, len(subsequence))
            else:
                max_length = max(max_length, len(subsequence))
        
        return max_length