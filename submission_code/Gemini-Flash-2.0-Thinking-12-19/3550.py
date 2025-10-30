import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        max_sum = -float('inf')

        row_indices_combinations = itertools.combinations(range(m), 3)
        col_indices_combinations = itertools.combinations(range(n), 3)

        for row_indices in row_indices_combinations:
            for col_indices in col_indices_combinations:
                col_permutations = itertools.permutations(col_indices)
                for p_cols in col_permutations:
                    current_sum = 0
                    for i in range(3):
                        current_sum += board[row_indices[i]][p_cols[i]]
                    max_sum = max(max_sum, current_sum)

        return max_sum