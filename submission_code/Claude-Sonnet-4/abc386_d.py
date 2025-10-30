Wait, this approach is getting too complex. Let me think of a simpler solution.

The key insight is that for a cell (i,j) to be black, we need both:
1. j ≤ r[i] (j is within the black prefix of row i)
2. i ≤ c[j] (i is within the black prefix of column j)

And we have the relationship: c[j] = max{i : r[i] ≥ j}