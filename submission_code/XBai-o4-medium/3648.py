from typing import List
from collections import defaultdict

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        if n == 0:
            return 0
        sum_s1 = 0
        for i in range(n):
            sum_s1 += fruits[i][i]
        
        # Initialize DP
        # prev_dp stores the maximum value for (j2, i3) at step k
        # At step 0, the children have made 0 moves, and are at their starting positions
        # The initial value includes the starting positions if they are not in S1 and not overlapping
        pos2 = (0, n-1)
        pos3 = (n-1, 0)
        initial_val = 0
        if pos2 != pos3:
            # Check if pos2 is not in S1
            if pos2[0] != pos2[1]:
                initial_val += fruits[pos2[0]][pos2[1]]
            # Check if pos3 is not in S1
            if pos3[0] != pos3[1]:
                initial_val += fruits[pos3[0]][pos3[1]]
        prev_dp = {}
        # The initial state is (j2, i3) = (n-1, n-1)
        j2_initial = n-1
        i3_initial = n-1
        prev_dp[(j2_initial, i3_initial)] = initial_val
        
        for k in range(0, n-1):
            curr_dp = {}
            for (j2, i3), prev_val in prev_dp.items():
                # Generate all possible moves for child 2 and child 3
                possible_j2_new = []
                if j2 - 1 >= 0:
                    possible_j2_new.append(j2 - 1)
                possible_j2_new.append(j2)
                if j2 + 1 < n:
                    possible_j2_new.append(j2 + 1)
                
                possible_i3_new = []
                if i3 - 1 >= 0:
                    possible_i3_new.append(i3 - 1)
                possible_i3_new.append(i3)
                if i3 + 1 < n:
                    possible_i3_new.append(i3 + 1)
                
                for j2_new in possible_j2_new:
                    for i3_new in possible_i3_new:
                        # new_k = k+1
                        new_k = k + 1
                        # Calculate incremental fruits for new positions
                        inc = 0
                        # Child 2's new position: (new_k, j2_new)
                        if j2_new != new_k:  # not in S1
                            inc += fruits[new_k][j2_new]
                        # Child 3's new position: (i3_new, new_k)
                        if i3_new != new_k:  # not in S1
                            inc += fruits[i3_new][new_k]
                        # Update curr_dp
                        key = (j2_new, i3_new)
                        if key in curr_dp:
                            if prev_val + inc > curr_dp[key]:
                                curr_dp[key] = prev_val + inc
                        else:
                            curr_dp[key] = prev_val + inc
            prev_dp = curr_dp
        
        max_s2s3 = max(prev_dp.values(), default=0)
        return sum_s1 + max_s2s3