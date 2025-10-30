class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        """
        We want to pick at most one cell from each row, with all chosen cell-values distinct,
        to maximize the sum of those values. This can be expressed as a maximum-weight bipartite
        matching problem between rows and possible values.

        Steps:
        1) We have up to 10 rows and values in [1..100].
        2) Build a 100x100 cost matrix where:
           - Rows 0..(R-1) represent the actual rows in the grid (R <= 10).
           - Columns 0..99 represent possible values 1..100.
           - cost[r][c] = -value if the grid row r (0-based) contains (c+1) somewhere
             (negate to use the Hungarian algorithm which finds a minimum-cost matching).
             If row r doesn't contain (c+1), cost[r][c] = 0 (so matching that column yields 0
             if we pick that value for that row, which effectively won't help).
           - For r >= R (extra padding rows up to 100), cost[r][c] = 0 so they don't contribute.
        3) Run the Hungarian algorithm (a.k.a. Kuhn-Munkres) to get the minimum total cost
           for matching each of the 100 rows with one of the 100 columns.
           However, we only truly need to match the first R rows. The algorithm will match all
           100 rows, but the additional 90 rows only yield 0 cost in any case.
        4) The negative of this minimum total cost is the maximum sum, since cost was negated.
           We only effectively match up to R = len(grid) rows (others produce zero).
        5) Return that result.

        This solves the problem within the given constraints (R <= 10, possible values up to 100).
        """

        # Collect basic dimensions
        R = len(grid)
        # For safety, handle small corner case if R=0
        if R == 0:
            return 0

        # We'll create a 100x100 cost matrix for the Hungarian algorithm
        N = 100  # For values from 1..100
        cost = [[0]*N for _ in range(N)]  # Initialize entire NxN with 0s

        # Precompute row-values presence
        # row_has_val[r][v] = True if grid[r] has value (v+1)
        row_has_val = [set() for _ in range(R)]
        for r in range(R):
            for val in grid[r]:
                row_has_val[r].add(val)

        # Fill the cost matrix (negate the value for maximum matching)
        for r in range(R):
            for v in range(1, 101):
                if v in row_has_val[r]:
                    cost[r][v-1] = -v

        # Hungarian (Kuhn-Munkres) Algorithm for min-cost perfect matching in NxN
        def hungarian_min_cost(matrix):
            """
            Implementation of the Hungarian algorithm (Kuhn-Munkres) to find a minimum-cost
            perfect matching. Expects an NxN matrix (list of lists). Returns the total
            matched cost.
            """
            n = len(matrix)
            u = [0]* (n+1)  # potential for 'left' side
            v = [0]* (n+1)  # potential for 'right' side
            p = [0]* (n+1)  # match for 'right' side
            way = [0]* (n+1)
            
            for i in range(1, n+1):
                p[0] = i
                j0 = 0
                minv = [float('inf')]*(n+1)
                used = [False]*(n+1)
                while True:
                    used[j0] = True
                    i0 = p[j0]  # row matched with column j0
                    j1 = 0
                    delta = float('inf')
                    for j in range(1, n+1):
                        if not used[j]:
                            cur = matrix[i0-1][j-1] - u[i0] - v[j]
                            if cur < minv[j]:
                                minv[j] = cur
                                way[j] = j0
                            if minv[j] < delta:
                                delta = minv[j]
                                j1 = j
                    for j in range(n+1):
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
            
            # p[j] = i means column j matched with row i
            # Calculate total cost of matching
            # matrix[i-1][j-1] is cost for i->j
            # matched pair is (i, j) for j in [1..n]
            res = 0
            for j in range(1, n+1):
                i = p[j]
                if i > 0:  # i=0 means no real row matched
                    res += matrix[i-1][j-1]
            return res

        min_cost = hungarian_min_cost(cost)
        # Convert from min_cost with negative edges to max sum
        return -min_cost