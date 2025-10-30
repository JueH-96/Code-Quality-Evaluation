Wait, this approach is too slow for N=30. Let me reconsider.

The key insight is that the number of fixed elements depends on the structure of row and column sums. Let me think more carefully:

1. If all row sums and column sums are in range [1, N-1], then no elements are fixed (we can permute)
2. If some rows/columns have sum 0 or N, those elements are fixed
3. The total number of fixed elements has specific patterns