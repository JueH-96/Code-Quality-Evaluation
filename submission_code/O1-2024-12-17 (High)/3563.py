from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # We'll solve this by formulating it as a maximum-weight bipartite matching problem
        # (also known as the Assignment problem). We have:
        #   - Rows of the matrix on one side (call them "row-nodes"),
        #   - Distinct values on the other side (call them "value-nodes"),
        #   - We can only match each row with at most one value, and each value can be used
        #     by at most one row.
        #
        # Our goal: maximize the sum of chosen values, subject to each row being matched
        # with at most one (unique) value. The Hungarian algorithm can handle "exactly one
        # match per row" by default, so to allow skipping rows, we add extra "dummy columns"
        # with cost 0. Matching a row with a dummy column corresponds to "not picking any
        # cell" from that row.
        #
        # Steps:
        # 1. Collect all distinct values from grid in a list distinct_vals.
        # 2. Let R = number of rows, M = number of distinct values.
        # 3. We construct a cost matrix of size N x N, where N = R + M.
        #    - The first R rows of this matrix correspond to the "actual rows" of grid.
        #    - The next M rows are "dummy rows" (0-cost, used to complete a perfect matching).
        #    - The first M columns correspond to the distinct values; columns M..(M+R-1)
        #      are dummy columns with cost=0 (used to represent skipping a row).
        # 4. For an actual row i and distinct value j, if that value appears in row i, we set
        #    cost[i][j] = -value (negative so that maximizing sum becomes minimizing cost).
        #    Otherwise, cost[i][j] = large positive (to forbid picking that value).
        # 5. For columns >= M, we set cost[i][col] = 0 (dummy columns).
        # 6. For dummy rows i >= R, we set cost[i][col] = 0 for all columns.
        # 7. We run the Hungarian algorithm on this cost matrix to find the minimal total cost.
        #    The result's negative is the maximum sum of distinct values with no two in the same row.
        #
        # Because R <= 10 and values in grid are up to 100, at most M <= 100.
        # Then N = R + M <= 110, so an O(N^3) Hungarian algorithm is feasible.
        
        # Step 1: Collect distinct values
        distinct_vals_set = set()
        for row in grid:
            distinct_vals_set.update(row)
        distinct_vals = list(distinct_vals_set)
        
        R = len(grid)
        M = len(distinct_vals)
        # N = R + M for the square cost matrix
        N = R + M
        
        # For fast checks, create a list of sets for each row
        row_sets = [set(r) for r in grid]
        
        # Step 2: Build the cost matrix of size N x N
        # We'll use a large positive number to represent invalid edges
        INF = 10**9
        cost = [[0]*N for _ in range(N)]
        
        # Fill the first R "real" rows vs. first M "value" columns
        for i in range(R):
            for j in range(M):
                val = distinct_vals[j]
                if val in row_sets[i]:
                    cost[i][j] = -val  # negative for maximizing
                else:
                    cost[i][j] = INF  # invalid edge
        
        # For real rows, columns >= M are dummy columns => cost=0
        for i in range(R):
            for j in range(M, N):
                cost[i][j] = 0
        
        # For dummy rows i >= R, cost=0 for all columns
        for i in range(R, N):
            for j in range(N):
                cost[i][j] = 0
        
        # Step 3: Hungarian algorithm (min-cost matching in an N x N matrix).
        #         We'll define a helper function to run it.
        
        def hungarian_minimize(matrix: List[List[int]]) -> (int, List[int]):
            # Implementation of the Hungarian (Kuhn-Munkres) algorithm for
            # finding a minimum-cost perfect matching in a bipartite graph.
            # matrix is NxN, cost[i][j] is the "cost" of matching row i with column j.
            
            n = len(matrix)
            u = [0] * (n + 1)  # potential for rows
            v = [0] * (n + 1)  # potential for columns
            p = [0] * (n + 1)  # which row is matched with column j
            way = [0] * (n + 1)
            
            for i in range(1, n + 1):
                p[0] = i
                j0 = 0
                minv = [float('inf')] * (n + 1)
                used = [False] * (n + 1)
                while True:
                    used[j0] = True
                    i0 = p[j0]
                    j1 = 0
                    delta = float('inf')
                    for j in range(1, n + 1):
                        if not used[j]:
                            cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                            if cur < minv[j]:
                                minv[j] = cur
                                way[j] = j0
                            if minv[j] < delta:
                                delta = minv[j]
                                j1 = j
                    for j in range(n + 1):
                        if used[j]:
                            u[p[j]] += delta
                            v[j] -= delta
                        else:
                            minv[j] -= delta
                    j0 = j1
                    if p[j0] == 0:
                        break
                while True:
                    j1 = way[j0]
                    p[j0] = p[j1]
                    j0 = j1
                    if j0 == 0:
                        break
            
            # p[j] = i means column j is matched with row i.
            # We want to invert that to get row->column notation.
            matching = [0] * (n + 1)
            for j in range(1, n + 1):
                matching[p[j]] = j
            
            # Calculate the minimal cost of this matching
            res = 0
            for i in range(1, n + 1):
                res += matrix[i - 1][matching[i] - 1]
            return res, matching
        
        # Step 4: Run Hungarian to get the minimal cost
        min_cost, matching = hungarian_minimize(cost)
        
        # Step 5: Our answer is the negative of that cost
        #         (because we stored valid edges as -value)
        max_sum = -min_cost
        
        # Since we must pick at least one cell if it's beneficial, the Hungarian
        # solution already does that (it won't skip all rows if any negative cost
        # match is available). If the matrix has no beneficial picks, the result
        # would be 0.
        
        return max_sum