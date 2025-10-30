class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        max_sum = -float('inf')

        for i1 in range(m):
            for j1 in range(n):
                for i2 in range(m):
                    for j2 in range(n):
                        if i1 == i2 or j1 == j2:
                            continue
                        for i3 in range(m):
                            for j3 in range(n):
                                if i1 == i3 or j1 == j3 or i2 == i3 or j2 == j3:
                                    continue
                                current_sum = board[i1][j1] + board[i2][j2] + board[i3][j3]
                                max_sum = max(max_sum, current_sum)
        return max_sum