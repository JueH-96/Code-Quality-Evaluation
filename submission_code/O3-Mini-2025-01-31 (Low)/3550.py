from typing import List
import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        # For each row, precompute a sorted list (max first) of candidate cells as (value, col).
        # Limit the number of candidates per row to L (e.g., 10)
        L = 10
        row_candidates = []
        for i in range(m):
            # create list of (value, col) pairs for row i
            cand = [(board[i][j], j) for j in range(n)]
            cand.sort(key=lambda x: x[0], reverse=True)
            if len(cand) > L:
                cand = cand[:L]
            row_candidates.append(cand)
        
        best = -10**18  # since board values can be as low as -1e9
        # iterate over all combinations of 3 distinct rows
        for r1, r2, r3 in itertools.combinations(range(m), 3):
            # For each of the three rows, try each candidate entry.
            for val1, c1 in row_candidates[r1]:
                for val2, c2 in row_candidates[r2]:
                    if c2 == c1:  # same column, skip
                        continue
                    for val3, c3 in row_candidates[r3]:
                        if c3 == c1 or c3 == c2:
                            continue
                        s = val1 + val2 + val3
                        if s > best:
                            best = s
        return best

# For local testing:
if __name__ == "__main__":
    sol = Solution()
    
    board1 = [[-3,1,1,1],[-3,1,-3,1],[-3,2,1,1]]
    print(sol.maximumValueSum(board1))  # Expected 4
    
    board2 = [[1,2,3],[4,5,6],[7,8,9]]
    print(sol.maximumValueSum(board2))  # Expected 15
    
    board3 = [[1,1,1],[1,1,1],[1,1,1]]
    print(sol.maximumValueSum(board3))  # Expected 3