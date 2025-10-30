class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # dp[i][j][k][lc][rl] = max length using i "AA", j "BB", k "AB",
        # ending with last_char=lc (0: none, 1:'A', 2:'B'), run_len=rl (0,1,2)
        dp = [[[[[-1]*3 for _ in range(3)] 
                 for _ in range(z+1)] 
                for _ in range(y+1)] 
               for _ in range(x+1)]
        dp[0][0][0][0][0] = 0
        
        ans = 0
        for i in range(x+1):
            for j in range(y+1):
                for k in range(z+1):
                    for lc in range(3):
                        for rl in range(3):
                            cur = dp[i][j][k][lc][rl]
                            if cur < 0:
                                continue
                            # update answer
                            if cur > ans:
                                ans = cur
                            
                            # try to add "AA"
                            if i < x:
                                # s1='A', s2='A'
                                # first A
                                if lc == 1:
                                    run1 = rl + 1
                                else:
                                    run1 = 1
                                if run1 < 3:
                                    # second A
                                    run2 = run1 + 1
                                    if run2 < 3:
                                        dp[i+1][j][k][1][run2] = max(
                                            dp[i+1][j][k][1][run2],
                                            cur + 2
                                        )
                            # try to add "BB"
                            if j < y:
                                # s1='B', s2='B'
                                if lc == 2:
                                    run1 = rl + 1
                                else:
                                    run1 = 1
                                if run1 < 3:
                                    run2 = run1 + 1
                                    if run2 < 3:
                                        dp[i][j+1][k][2][run2] = max(
                                            dp[i][j+1][k][2][run2],
                                            cur + 2
                                        )
                            # try to add "AB"
                            if k < z:
                                # s1='A'
                                if lc == 1:
                                    run1 = rl + 1
                                else:
                                    run1 = 1
                                if run1 < 3:
                                    # s2='B'
                                    run2 = 1  # switches from A to B
                                    dp[i][j][k+1][2][run2] = max(
                                        dp[i][j][k+1][2][run2],
                                        cur + 2
                                    )
        return ans