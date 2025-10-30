from typing import List
import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        # For each column j, we want to extract L best candidates (row, value)
        # (We choose L = min(m, 3), because we only need 3 distinct rows.)
        L = 3
        # candidate[j] will be a list of tuples (row, value) sorted descending by value.
        candidate = [None] * n
        for j in range(n):
            col_candidates = []
            for i in range(m):
                col_candidates.append((i, board[i][j]))
            # sort descending by value
            col_candidates.sort(key=lambda x: x[1], reverse=True)
            # keep only top L candidates (if there are fewer than L, take all)
            candidate[j] = col_candidates[:L]
        
        best = -10**20  # a value lower than -1e9*3
        # Iterate over all triples of distinct columns (in sorted order)
        for j1, j2, j3 in itertools.combinations(range(n), 3):
            # Quick upper-bound: even if we ignore the row restrictions, the best sum
            # we can get from these three columns is the sum of the top cell in each.
            ub = candidate[j1][0][1] + candidate[j2][0][1] + candidate[j3][0][1]
            if ub <= best:
                continue
            
            # For each candidate from column j1.
            for r1, v1 in candidate[j1]:
                # Since candidate[j1] is sorted descending by value, if even the best
                # from j2 and j3 added to v1 cannot beat current best, we can break.
                if v1 + candidate[j2][0][1] + candidate[j3][0][1] <= best:
                    break
                for r2, v2 in candidate[j2]:
                    if r2 == r1:
                        continue   # same row conflict
                    sum12 = v1 + v2
                    if sum12 + candidate[j3][0][1] <= best:
                        # candidate[j2] is sorted, so later ones give lower v2.
                        break
                    for r3, v3 in candidate[j3]:
                        if r3 == r1 or r3 == r2:
                            continue
                        total = sum12 + v3
                        if total > best:
                            best = total
                        # Since candidate[j3] is sorted descending, the first valid candidate
                        # gives the highest possible value for column j3 for this fixed (r1, r2)
                        break  # break the innermost loop over candidate[j3]
        return best