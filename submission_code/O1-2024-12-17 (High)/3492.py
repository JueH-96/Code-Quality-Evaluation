class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        """
        We are to count the number of submatrices that:
          1) contain the cell (0,0),
          2) have an equal number of 'X' and 'Y',
          3) and contain at least one 'X'.

        From the examples (especially Example 2), it becomes clear that
        "contain grid[0][0]" means the submatrix must include the top-left
        cell (0,0). That is, if we pick a submatrix with corners
        (0,0) --> (r,c), it includes (0,0). We want to know how many
        of these "anchored-at-(0,0)" submatrices satisfy #X = #Y and #X > 0.

        In other words, for each (r, c), consider the submatrix from (0,0)
        down to (r,c). Let #X(r,c) = number of 'X' there, and #Y(r,c) = number of 'Y'.
        We need #X(r,c) = #Y(r,c) and #X(r,c) > 0.

        We can do this efficiently by building 2D prefix sums for X and Y:

            Xps[r][c] = number of 'X' in submatrix (0..r, 0..c)
            Yps[r][c] = number of 'Y' in submatrix (0..r, 0..c)

        Then #X(0..r,0..c) = Xps[r][c]
             #Y(0..r,0..c) = Yps[r][c]

        We simply check:
            if Xps[r][c] == Yps[r][c] and Xps[r][c] > 0:
                answer += 1

        This gives an O(N*M) solution, which is efficient for grids up to 1000 x 1000.
        """

        n = len(grid)
        m = len(grid[0]) if n > 0 else 0

        # Build 2D prefix sums for counts of 'X' and 'Y'
        Xps = [[0]*m for _ in range(n)]  # X prefix sums
        Yps = [[0]*m for _ in range(n)]  # Y prefix sums

        for r in range(n):
            for c in range(m):
                # How many X's / Y's contributed by this cell
                addX = 1 if grid[r][c] == 'X' else 0
                addY = 1 if grid[r][c] == 'Y' else 0

                # Prefix sums: standard inclusion-exclusion
                aboveX = Xps[r-1][c] if r > 0 else 0
                leftX  = Xps[r][c-1] if c > 0 else 0
                diagX  = Xps[r-1][c-1] if (r > 0 and c > 0) else 0

                aboveY = Yps[r-1][c] if r > 0 else 0
                leftY  = Yps[r][c-1] if c > 0 else 0
                diagY  = Yps[r-1][c-1] if (r > 0 and c > 0) else 0

                Xps[r][c] = aboveX + leftX - diagX + addX
                Yps[r][c] = aboveY + leftY - diagY + addY

        answer = 0
        # Check each (r, c) for #X == #Y > 0 in submatrix (0..r, 0..c)
        for r in range(n):
            for c in range(m):
                if Xps[r][c] == Yps[r][c] and Xps[r][c] > 0:
                    answer += 1

        return answer