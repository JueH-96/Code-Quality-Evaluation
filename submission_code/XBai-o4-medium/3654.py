from typing import List

class Solution:
    def minArraySum(self, nums: List[int], k: int, op1: int, op2: int) -> int:
        original_sum = sum(nums)
        m, n = op1 + 1, op2 + 1
        
        # Initialize DP table with -infinity
        prev = [[float('-inf')] * n for _ in range(m)]
        prev[0][0] = 0  # Starting with 0 operations used and 0 reduction
        
        for num in nums:
            current = [[float('-inf')] * n for _ in range(m)]
            for a in range(m):
                for b in range(n):
                    if prev[a][b] == float('-inf'):
                        continue
                    # Option A: do nothing
                    if current[a][b] < prev[a][b]:
                        current[a][b] = prev[a][b]
                    # Option B: apply op1
                    if a + 1 < m:
                        reduction_b = num - ((num + 1) // 2)
                        if current[a+1][b] < prev[a][b] + reduction_b:
                            current[a+1][b] = prev[a][b] + reduction_b
                    # Option C: apply op2
                    if b + 1 < n and num >= k:
                        reduction_c = k
                        if current[a][b+1] < prev[a][b] + reduction_c:
                            current[a][b+1] = prev[a][b] + reduction_c
                    # Option D: apply both op1 and op2
                    if a + 1 < m and b + 1 < n:
                        ceil_half = (num + 1) // 2
                        if ceil_half >= k:
                            reduction_d = (num - ceil_half) + k
                            if current[a+1][b+1] < prev[a][b] + reduction_d:
                                current[a+1][b+1] = prev[a][b] + reduction_d
            prev = current
        
        max_reduction = 0
        for a in range(m):
            for b in range(n):
                if prev[a][b] > max_reduction:
                    max_reduction = prev[a][b]
        return original_sum - max_reduction