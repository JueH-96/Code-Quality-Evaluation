import math

class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        pos = [i for i in range(len(s1)) if s1[i] != s2[i]]
        len_pos = len(pos)
        if len_pos % 2 == 1:
            return -1
        if len_pos == 0:
            return 0
        m = len_pos
        p = pos
        INF = 10**9 + 5
        dp = [[0 for _ in range(m)] for _ in range(m)]
        
        for len_diff in range(0, m):
            for i in range(m):
                j = i + len_diff
                if j >= m:
                    continue
                num = j - i + 1
                if num % 2 == 1:
                    dp[i][j] = INF
                else:
                    min_cost = INF
                    for k in range(i, j + 1):
                        if k == i:
                            continue
                        if abs(k - i) == 1:
                            dist_pos = abs(p[i] - p[k])
                            cost_pair = min(dist_pos, x)
                        else:
                            cost_pair = x
                        # subcost left
                        a_left = i + 1
                        b_left = k - 1
                        if a_left > b_left:
                            left_cost = 0
                        else:
                            num_left = b_left - a_left + 1
                            if num_left % 2 == 1:
                                left_cost = INF
                            else:
                                left_cost = dp[a_left][b_left]
                        # subcost right
                        a_right = k + 1
                        b_right = j
                        if a_right > b_right:
                            right_cost = 0
                        else:
                            num_right = b_right - a_right + 1
                            if num_right % 2 == 1:
                                right_cost = INF
                            else:
                                right_cost = dp[a_right][b_right]
                        total_cost = cost_pair + left_cost + right_cost
                        if total_cost < min_cost:
                            min_cost = total_cost
                    dp[i][j] = min_cost
        
        return dp[0][m - 1]