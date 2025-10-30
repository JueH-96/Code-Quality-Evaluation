from itertools import combinations
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0]) if m > 0 else 0
        max_sum = -float('inf')
        
        # Iterate over all combinations of 3 distinct rows
        for rows in combinations(range(m), 3):
            r0, r1, r2 = board[rows[0]], board[rows[1]], board[rows[2]]
            dp = [-float('inf')] * 8
            dp[0] = 0  # Initial state: no rows selected
            
            for col in range(n):
                new_dp = dp.copy()
                for state in range(8):
                    if dp[state] == -float('inf'):
                        continue
                    # Check each of the three rows (0, 1, 2) for possible transitions
                    for row in range(3):
                        if not (state & (1 << row)):
                            new_state = state | (1 << row)
                            val = [r0[col], r1[col], r2[col]][row]
                            if new_dp[new_state] < dp[state] + val:
                                new_dp[new_state] = dp[state] + val
                dp = new_dp
            
            if dp[7] > max_sum:
                max_sum = dp[7]
        
        return max_sum