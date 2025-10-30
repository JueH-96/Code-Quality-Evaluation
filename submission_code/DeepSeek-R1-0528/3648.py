from typing import List
from collections import defaultdict

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        dp = defaultdict(int)
        start_value = fruits[0][0] + fruits[0][n-1] + fruits[n-1][0]
        if (0, 0) == (0, n-1):
            start_value -= fruits[0][0]
        if (0, 0) == (n-1, 0):
            start_value -= fruits[0][0]
        if (0, n-1) == (n-1, 0):
            start_value -= fruits[0][n-1]
        if (0, 0) == (0, n-1) == (n-1, 0):
            start_value = fruits[0][0]
        dp[(0, 0, n-1, n-1)] = start_value
        
        moves1 = [(1, 0), (0, 1), (1, 1)]
        moves2 = [-1, 0, 1]
        moves3 = [-1, 0, 1]
        
        for k in range(0, n-1):
            next_dp = defaultdict(int)
            for state, total in dp.items():
                i1, j1, a, b = state
                for dx, dy in moves1:
                    i1_next = i1 + dx
                    j1_next = j1 + dy
                    if i1_next < 0 or i1_next >= n or j1_next < 0 or j1_next >= n:
                        continue
                    if i1_next > k+1 or j1_next > k+1:
                        continue
                    if i1_next + j1_next < k+1 or i1_next + j1_next > 2*(k+1):
                        continue
                    for da in moves2:
                        a_next = a + da
                        if a_next < 0 or a_next >= n:
                            continue
                        for db in moves3:
                            b_next = b + db
                            if b_next < 0 or b_next >= n:
                                continue
                            rooms = set()
                            rooms.add((i1_next, j1_next))
                            rooms.add((k+1, a_next))
                            rooms.add((b_next, k+1))
                            added_value = sum(fruits[i][j] for i, j in rooms)
                            new_state = (i1_next, j1_next, a_next, b_next)
                            new_total = total + added_value
                            if new_total > next_dp[new_state]:
                                next_dp[new_state] = new_total
            dp = next_dp
        
        final_state = (n-1, n-1, n-1, n-1)
        return dp.get(final_state, 0)