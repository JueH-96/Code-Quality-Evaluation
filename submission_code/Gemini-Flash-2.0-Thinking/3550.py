from itertools import combinations, permutations

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        max_sum = float('-inf')

        for row_indices in combinations(range(m), 3):
            for col_indices in combinations(range(n), 3):
                for col_perm in permutations(col_indices):
                    current_sum = board[row_indices[0]][col_perm[0]] + \
                                  board[row_indices[1]][col_perm[1]] + \
                                  board[row_indices[2]][col_perm[2]]
                    max_sum = max(max_sum, current_sum)

        return max_sum