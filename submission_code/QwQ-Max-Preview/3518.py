from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        if n < 4:
            return 0  # According to constraints, n >=4
        
        # Initialize for k=0 (first element in a)
        prev_prefix = [-float('inf')] * n
        prev_prefix[0] = a[0] * b[0]
        for i in range(1, n):
            prev_prefix[i] = max(prev_prefix[i-1], a[0] * b[i])
        
        # Iterate for the remaining elements in a (k=1, 2, 3)
        for k in range(1, 4):
            current_dp = [-float('inf')] * n
            current_prefix = [-float('inf')] * n
            current_max = -float('inf')
            for i in range(k, n):
                # The maximum value from previous step up to i-1
                prev_max = prev_prefix[i-1] if i-1 >= 0 else -float('inf')
                current_dp_i = a[k] * b[i] + prev_max
                current_dp[i] = current_dp_i
                # Update current_max and current_prefix
                if current_dp_i > current_max:
                    current_max = current_dp_i
                current_prefix[i] = current_max
            # Update prev_prefix for the next iteration
            prev_prefix = current_prefix
        
        # The maximum score is the maximum value in prev_prefix starting from index 3
        return max(prev_prefix[3:])