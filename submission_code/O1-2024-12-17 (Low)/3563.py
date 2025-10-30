class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        """
        We need to pick at most one cell per row, with all chosen values distinct,
        so as to maximize the sum of chosen values.

        This can be solved by building a bipartite graph:
          - One part represents rows (up to N=10).
          - The other part represents all distinct values in the grid (up to 100 distinct values).
          - An edge from row i to value v exists if v is in row i. The 'weight' of that edge is v.

        We then want the maximum-sum matching in this bipartite graph.  But the standard Hungarian
        (Kuhn-Munkres) algorithm is formulated for an N×N cost (or weight) matrix.  We handle
        the rectangular case by padding to D = max(N, number_of_distinct_values) on both dimensions
        and using a "min-cost" assignment version of the Hungarian algorithm, after converting
        the weights (values) into costs = (max_value + 1) - value if row i has that value,
        or (max_value + 1) if not present in row i.  Rows or columns beyond the real data
        are dummy and assigned cost=0 so they don't affect the matching if used to skip.

        After the Hungarian algorithm, we invert the matched costs to get the sum of the selected
        values.  This yields the maximum possible sum under the constraints.

        Time complexity is O(D^3) with D ≤ 100, which is fine for our constraints.
        """

        # Collect distinct values.
        distinct_vals_set = set()
        for row in grid:
            distinct_vals_set.update(row)
        distinct_vals = sorted(distinct_vals_set)
        max_val = max(distinct_vals)  # We'll need this for cost conversion

        n = len(grid)                  # number of rows
        v = len(distinct_vals)         # number of distinct values
        D = max(n, v)                  # dimension for the Hungarian

        # Build the cost matrix of size D x D.
        # cost[i][j] will be the "cost" of matching row i to distinct value j.
        # If row i does not contain distinct_vals[j], set cost high (max_val+1).
        # Otherwise cost = (max_val+1 - that_value).
        # Rows above n or columns above v are dummy => cost=0.
        cost = [[0] * D for _ in range(D)]
        for i in range(n):
            rowset = set(grid[i])
            for j in range(v):
                val = distinct_vals[j]
                if val in rowset:
                    cost[i][j] = (max_val + 1) - val
                else:
                    cost[i][j] = (max_val + 1)
        # The extra rows or columns get cost=0; they are already 0 by default.

        # Hungarian (Kuhn-Munkres) for Min-Cost Perfect Matching in a D x D matrix.
        # Returns (minimum_cost, match_row, match_col).
        # match_row[i] = column matched with row i
        # match_col[j] = row matched with column j
        def hungarian_min_cost(matrix: List[List[int]]) -> (int, List[int], List[int]):
            # Adapted from a classic Hungarian/ Kuhn-Munkres implementation.
            # Complexity: O(D^3).
            N = len(matrix)
            u = [0] * (N+1)
            v = [0] * (N+1)
            p = [0] * (N+1)
            way = [0] * (N+1)

            for i in range(1, N+1):
                p[0] = i
                j0 = 0
                minv = [float('inf')] * (N+1)
                used = [False] * (N+1)
                while True:
                    used[j0] = True
                    i0 = p[j0]
                    j1 = 0
                    delta = float('inf')
                    for j in range(1, N+1):
                        if not used[j]:
                            cur = matrix[i0-1][j-1] - u[i0] - v[j]
                            if cur < minv[j]:
                                minv[j] = cur
                                way[j] = j0
                            if minv[j] < delta:
                                delta = minv[j]
                                j1 = j
                    for j in range(N+1):
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

            # p[j] = the row matched to column j
            # Reconstruct the matching cost
            match_row = [0]*N
            match_col = [0]*N
            for j in range(1, N+1):
                match_row[p[j]-1] = j-1
            for j in range(N):
                match_col[j] = p[j+1]-1

            # minimum cost is -v[0], but we can also compute directly
            min_cost = -v[0]
            for i in range(1, N+1):
                min_cost += matrix[i-1][match_row[i-1]]
            return min_cost, match_row, match_col

        # Run Hungarian on our cost matrix
        min_cost, match_row, match_col = hungarian_min_cost(cost)

        # Now match_row[i] = j means row i matched with column j.
        # We only care about i < n, j < v for real row/value pairs.
        # The cost contributed there is cost[i][j]. The corresponding value is:
        # original_value = (max_val + 1) - cost[i][j], if cost[i][j] != (max_val+1).
        # But we only add it if j < v (not a dummy column).
        # Also we must check that the row actually had that value (cost check).
        # A simpler method: for each row i < n, if j < v, we add distinct_vals[j].
        # Because we constructed cost so that if row i doesn't have distinct_vals[j],
        # cost is (max_val+1), which is a "high cost"; the matching might sometimes
        # pick that if it's beneficial, but that would yield 0 value for us. So we
        # can check if cost[i][j] < (max_val+1), i.e., it was a real edge.
        
        ans = 0
        for i in range(n):
            j = match_row[i]
            # If j < v and cost[i][j] < (max_val+1), we matched row i with a real value j
            if j < v and cost[i][j] < (max_val + 1):
                ans += distinct_vals[j]

        return ans