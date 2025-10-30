from typing import List

class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        # n is even, let k = n // 2 be the number of mirror-pairs (rungs of the ladder)
        k = n // 2
        INF = 10**30
        
        # dp[a][b]: minimum cost up to pair p-1, where the color of the "left" node in pair p-1 is a,
        # and the color of the "right" (mirrored) node is b; with the invariant a != b.
        dp = [[INF]*3 for _ in range(3)]
        
        # Initialize for p = 0, which pairs house 0 and house n-1
        for a in range(3):
            for b in range(3):
                if a != b:
                    dp[a][b] = cost[0][a] + cost[n-1][b]
        
        # Iterate over pairs p = 1..k-1
        for p in range(1, k):
            dp2 = [[INF]*3 for _ in range(3)]
            left_idx = p
            right_idx = n - 1 - p
            # try all previous (a0,b0) and current (a1,b1)
            for a0 in range(3):
                for b0 in range(3):
                    prev_val = dp[a0][b0]
                    if prev_val >= INF:
                        continue
                    # current colors
                    for a1 in range(3):
                        if a1 == a0:  # adjacent on left rail
                            continue
                        for b1 in range(3):
                            if b1 == b0:  # adjacent on right rail
                                continue
                            if a1 == b1:  # same color in a mirror-pair
                                continue
                            c = prev_val + cost[left_idx][a1] + cost[right_idx][b1]
                            if c < dp2[a1][b1]:
                                dp2[a1][b1] = c
            dp = dp2
        
        # answer is the minimum over final dp[a][b] with a != b
        ans = INF
        for a in range(3):
            for b in range(3):
                if a != b and dp[a][b] < ans:
                    ans = dp[a][b]
        return ans