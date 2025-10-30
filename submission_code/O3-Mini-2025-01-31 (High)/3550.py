from typing import List
import sys

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        # Quick check: if every cell has the same value, then every valid
        # placement gives the same sum.
        first_val = board[0][0]
        uniform = True
        for i in range(m):
            for j in range(n):
                if board[i][j] != first_val:
                    uniform = False
                    break
            if not uniform:
                break
        if uniform:
            return 3 * first_val

        # Build a list of all cells as tuples: (value, row, col)
        cells = []
        for i in range(m):
            for j in range(n):
                cells.append((board[i][j], i, j))
        # Sort cells in descending order by value.
        cells.sort(key=lambda x: x[0], reverse=True)
        
        # Get a greedy solution:
        # (Greedily “pick” cells from the sorted list that do not share a row or column.)
        greedy_sum = 0
        used_rows = set()
        used_cols = set()
        count = 0
        for val, r, c in cells:
            if r not in used_rows and c not in used_cols:
                greedy_sum += val
                used_rows.add(r)
                used_cols.add(c)
                count += 1
                if count == 3:
                    break
        best_result = greedy_sum if count == 3 else -10**18

        # Set a high recursion limit (since worst-case board size is 100x100 --> 10^4 cells)
        sys.setrecursionlimit(1000000)
        
        # Use branch‐and–bound recursion. At each step we decide whether to pick a cell 
        # (if its row and col are “free”) and we prune if even the best possible extension
        # (looking at the next few cells in sorted order) cannot beat the current best result.
        def search(index, count, current_sum, used_rows, used_cols):
            nonlocal best_result
            # Base–case: we have selected 3 cells.
            if count == 3:
                if current_sum > best_result:
                    best_result = current_sum
                return
            if index >= len(cells):
                return
            remaining = 3 - count
            # Use a tight (yet fast) upper bound:
            # (Since the list “cells” is sorted, the next ‘remaining’ cells all have value at most cells[index][0]. 
            # A slightly tighter bound is the sum of the next 'remaining' cell values.)
            if index + remaining <= len(cells):
                ub = current_sum + sum(cells[i][0] for i in range(index, index+remaining))
            else:
                ub = current_sum
            if ub <= best_result:
                return
            
            # Try every remaining candidate starting at position "index".
            for i in range(index, len(cells)):
                val, r, c = cells[i]
                # Skip if the cell’s row or column is already used.
                if r in used_rows or c in used_cols:
                    continue
                # Choose this cell
                used_rows.add(r)
                used_cols.add(c)
                search(i+1, count+1, current_sum+val, used_rows, used_cols)
                used_rows.remove(r)
                used_cols.remove(c)
        
        search(0, 0, 0, set(), set())
        return best_result