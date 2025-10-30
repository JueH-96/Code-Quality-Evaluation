from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # Map each distinct value to a unique bit position
        all_vals = set()
        for row in grid:
            all_vals.update(row)
        val_list = sorted(all_vals)
        val_to_bit = {v: i for i, v in enumerate(val_list)}
        
        # Convert each row into list of (value, bitmask) and compute row max
        rows = []
        for row in grid:
            cells = [(v, 1 << val_to_bit[v]) for v in row]
            maxv = max(v for v, _ in cells)
            rows.append((maxv, cells))
        
        # We can reorder rows for better pruning: sort descending by row-max
        rows.sort(key=lambda x: -x[0])
        # Build a list of remaining-max prefix sums for pruning
        n = len(rows)
        rem_max = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            rem_max[i] = rem_max[i + 1] + rows[i][0]
        
        self.best = 0
        
        def dfs(i: int, used_mask: int, curr_sum: int):
            # Prune if even taking all remaining max can't beat best
            if curr_sum + rem_max[i] <= self.best:
                return
            if i == n:
                if curr_sum > self.best:
                    self.best = curr_sum
                return
            # Option 1: skip this row
            dfs(i + 1, used_mask, curr_sum)
            # Option 2: pick one cell in this row if value not used
            _, cells = rows[i]
            for v, b in cells:
                if not (used_mask & b):
                    dfs(i + 1, used_mask | b, curr_sum + v)
        
        dfs(0, 0, 0)
        return self.best


# Example runs:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxScore([[1,2,3],[4,3,2],[1,1,1]]))  # Expected 8
    print(sol.maxScore([[8,7,6],[8,3,2]]))          # Expected 15