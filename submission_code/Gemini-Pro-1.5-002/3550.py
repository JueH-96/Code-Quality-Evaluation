class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = -float('inf')

        for i in range(m):
            for j in range(n):
                for k in range(m):
                    for l in range(n):
                        for p in range(m):
                            for q in range(n):
                                if i != k and i != p and j != l and j != q and k != p and l != q:
                                    max_sum = max(max_sum, board[i][j] + board[k][l] + board[p][q])

        return max_sum