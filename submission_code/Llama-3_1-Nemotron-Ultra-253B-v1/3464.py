from typing import List

class Solution:
    def maximumTotalCost(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # Compute the prefix sum S of the alt array
        S = [0] * (n + 1)
        for i in range(1, n + 1):
            sign = 1 if (i - 1) % 2 == 0 else -1
            S[i] = S[i - 1] + nums[i - 1] * sign
        
        max_even = 0  # Initially, j=0 is even, A[0] = 0
        max_odd = -float('inf')
        
        for i in range(1, n + 1):
            current_S = S[i]
            # Calculate the current dp_i
            option1 = max_even + current_S
            option2 = max_odd - current_S
            dp_i = max(option1, option2)
            
            # Update max_even or max_odd based on current i's parity
            sign_i = 1 if i % 2 == 0 else -1
            A_i = dp_i - sign_i * current_S
            
            if i % 2 == 0:
                max_even = max(max_even, A_i)
            else:
                max_odd = max(max_odd, A_i)
        
        return dp_i