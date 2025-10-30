from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        dp_prev = [a[0] * num for num in b]
        
        for k in range(2, 5):  # k=2,3,4
            dp_current = [0] * n
            low = k - 2
            start_i = k - 1
            max_prev_val = dp_prev[low]
            for i in range(start_i, n):
                dp_current[i] = a[k-1] * b[i] + max_prev_val
                max_prev_val = max(max_prev_val, dp_prev[i])
            dp_prev = dp_current
        
        # dp_prev now holds dp for k=4
        return max(dp_prev[i] for i in range(3, n))