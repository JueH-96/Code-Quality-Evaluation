class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 10**9 + 7
        
        def sum_of_distances(length):
            # Calculate the sum of distances for a 1D line of given length
            # Sum of distances between all pairs in a line of length L is:
            # (1*(L-1) + 2*(L-2) + ... + (L-1)*1)
            # This can be simplified to sum(i*(L-i) for i in range(1, L))
            total = 0
            for i in range(1, length):
                total += i * (length - i)
                total %= MOD
            return total
        
        # Calculate the sum of distances for all rows (1D problem for m rows)
        total_row_distances = sum_of_distances(m)
        # Calculate the sum of distances for all columns (1D problem for n columns)
        total_col_distances = sum_of_distances(n)
        
        # Each pair of pieces contributes to the total sum of distances
        # The number of ways to choose 2 pieces out of k is C(k, 2) = k*(k-1)/2
        # Each row contributes to the row distance and each column contributes to the column distance
        num_pairs = k * (k - 1) // 2
        
        # Total distance is the sum of all row distances and all column distances
        # multiplied by the number of pairs
        result = (num_pairs * (total_row_distances + total_col_distances)) % MOD
        
        return result