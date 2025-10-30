from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        # Child1's path is forced: (0,0), (1,1), ..., (n-1,n-1)
        diag = 0
        for i in range(n):
            diag += fruits[i][i]
        
        NEG = -10**9  # a number lower than any possible valid collected sum
        
        # For Child2:
        # Child2 starts at (0, n-1) and at time t is at (t, j) where j is chosen.
        # Allowed transitions for j: j -> j-1, j, j+1.
        # Reward at time t when at column j is fruits[t][j] except if j equals t (diagonal cell)
        # because the fruit there is already collected by Child1.
        dp2 = [NEG] * n  # dp2[j] represents best reward from time t for state j.
        # At time 0, Child2 is at (0, n-1); since n-1 != 0 for n>=2, we count that cell.
        dp2[n - 1] = fruits[0][n - 1]
        for t in range(0, n - 1):
            new_dp2 = [NEG] * n
            # For each possible column state j at time t
            for j in range(n):
                curr = dp2[j]
                if curr == NEG:
                    continue
                # Try moves: j-1, j, j+1
                # next time is t+1, and next cell will be (t+1, nj)
                # Only add the fruit from that room if nj != t+1 (i.e. not on the diagonal)
                nj = j - 1
                if nj >= 0:
                    add = 0 if (nj == t + 1) else fruits[t + 1][nj]
                    candidate = curr + add
                    if candidate > new_dp2[nj]:
                        new_dp2[nj] = candidate
                nj = j
                # j itself:
                add = 0 if (j == t + 1) else fruits[t + 1][j]
                candidate = curr + add
                if candidate > new_dp2[j]:
                    new_dp2[j] = candidate
                nj = j + 1
                if nj < n:
                    add = 0 if (nj == t + 1) else fruits[t + 1][nj]
                    candidate = curr + add
                    if candidate > new_dp2[nj]:
                        new_dp2[nj] = candidate
            dp2 = new_dp2
        # At time n-1, Child2 must be at (n-1, n-1)
        answer2 = dp2[n - 1]
        
        # For Child3:
        # Child3 starts at (n-1, 0) and at time t is at (i, t) where i is chosen.
        # Allowed transitions for i: i -> i-1, i, i+1.
        # Reward at time t when at row i is fruits[i][t] except if i == t (diagonal).
        dp3 = [NEG] * n  # dp3[i] will represent best reward for being at row i at time t.
        # Start: time 0, Child3 is at (n-1, 0). For n>=2, n-1 != 0 so add cell's fruit.
        dp3[n - 1] = fruits[n - 1][0]
        for t in range(0, n - 1):
            new_dp3 = [NEG] * n
            for i in range(n):
                curr = dp3[i]
                if curr == NEG:
                    continue
                # Try transitions: i-1, i, i+1
                ni = i - 1
                if ni >= 0:
                    add = 0 if (ni == t + 1) else fruits[ni][t + 1]
                    candidate = curr + add
                    if candidate > new_dp3[ni]:
                        new_dp3[ni] = candidate
                ni = i
                add = 0 if (i == t + 1) else fruits[i][t + 1]
                candidate = curr + add
                if candidate > new_dp3[i]:
                    new_dp3[i] = candidate
                ni = i + 1
                if ni < n:
                    add = 0 if (ni == t + 1) else fruits[ni][t + 1]
                    candidate = curr + add
                    if candidate > new_dp3[ni]:
                        new_dp3[ni] = candidate
            dp3 = new_dp3
        # At time n-1, Child3 must be at (n-1, n-1)
        answer3 = dp3[n - 1]
        
        # Total maximum fruits is sum of fruits collected along each chosen room in union.
        # Since Child1 collects all diagonal fruits, and the rewards for Child2 and Child3
        # have been set to zero when on the diagonal, we can simply add the three parts.
        return diag + answer2 + answer3