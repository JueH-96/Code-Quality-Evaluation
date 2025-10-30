from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        # DP for P2
        DP2 = [[0 for _ in range(n)] for _ in range(n)]
        prev_j2 = [[-1 for _ in range(n)] for _ in range(n)]

        # Initialize i=0 for P2
        for j in range(n):
            if j == n - 1:
                DP2[0][j] = 0 if 0 == j else fruits[0][j]
            else:
                DP2[0][j] = float('-inf')

        for i in range(1, n):
            for cur_j in range(n):
                val = 0 if i == cur_j else fruits[i][cur_j]
                max_val = float('-inf')
                best_prev_j = -1
                for d in [-1, 0, 1]:
                    p_j = cur_j + d
                    if 0 <= p_j < n:
                        if DP2[i - 1][p_j] > max_val:
                            max_val = DP2[i - 1][p_j]
                            best_prev_j = p_j
                if max_val > float('-inf'):
                    DP2[i][cur_j] = val + max_val
                    prev_j2[i][cur_j] = best_prev_j
                else:
                    DP2[i][cur_j] = float('-inf')
                    prev_j2[i][cur_j] = -1

        # Backtrack P2 path from (n-1, n-1)
        P2_path = []
        current_i = n - 1
        current_j = n - 1
        while True:
            P2_path.append((current_i, current_j))
            if current_i == 0:
                break
            prev_j = prev_j2[current_i][current_j]
            current_i -= 1
            current_j = prev_j
        P2_path.reverse()

        # DP for P3
        DP3 = [[0 for _ in range(n)] for _ in range(n)]  # DP[j][i]
        prev_i3 = [[-1 for _ in range(n)] for _ in range(n)]

        # Initialize j=0 for P3
        for i in range(n):
            if i == n - 1:
                DP3[0][i] = 0 if 0 == i else fruits[i][0]  # for cell (i,0)
            else:
                DP3[0][i] = float('-inf')

        for j in range(1, n):
            for cur_i in range(n):
                val = 0 if cur_i == j else fruits[cur_i][j]  # for cell (cur_i, j)
                max_val = float('-inf')
                best_prev_i = -1
                for d in [-1, 0, 1]:
                    p_i = cur_i + d  # prev i at j-1
                    if 0 <= p_i < n:
                        if DP3[j - 1][p_i] > max_val:
                            max_val = DP3[j - 1][p_i]
                            best_prev_i = p_i
                if max_val > float('-inf'):
                    DP3[j][cur_i] = val + max_val
                    prev_i3[j][cur_i] = best_prev_i
                else:
                    DP3[j][cur_i] = float('-inf')
                    prev_i3[j][cur_i] = -1

        # Backtrack P3 path from j=n-1, i=n-1
        P3_path = []
        current_j = n - 1
        current_i = n - 1
        while True:
            P3_path.append((current_i, current_j))
            if current_j == 0:
                break
            prev_i = prev_i3[current_j][current_i]
            current_j -= 1
            current_i = prev_i
        P3_path.reverse()

        # P1 path
        P1_path = [(i, i) for i in range(n)]

        # Union of all cells
        cell_set = set(P1_path) | set(P2_path) | set(P3_path)

        # Sum the fruits
        total_sum = 0
        for row, col in cell_set:
            total_sum += fruits[row][col]

        return total_sum