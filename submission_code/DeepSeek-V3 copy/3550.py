from itertools import permutations

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = -float('inf')
        
        # Generate all possible combinations of 3 rows and 3 columns
        for rows in permutations(range(m), 3):
            for cols in permutations(range(n), 3):
                # Ensure that the selected cells are not in the same row or column
                if len(set(rows)) == 3 and len(set(cols)) == 3:
                    current_sum = 0
                    for i in range(3):
                        current_sum += board[rows[i]][cols[i]]
                    if current_sum > max_sum:
                        max_sum = current_sum
        return max_sum