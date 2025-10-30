class Solution:
    def distanceSum(self, m: int, n: int, k: int) -> int:
        MOD = 1000000007
        
        def nCr(n, r):
            if r > n:
                return 0
            r = min(r, n-r)
            num = den = 1
            for i in range(r):
                num = (num * (n - i)) % MOD
                den = (den * (i + 1)) % MOD
            # Fermat's little theorem for modular multiplicative inverse
            return (num * pow(den, MOD-2, MOD)) % MOD
        
        total = 0
        # For each pair of cells in the grid
        for x1 in range(m):
            for y1 in range(n):
                for x2 in range(m):
                    for y2 in range(n):
                        if x1 == x2 and y1 == y2:
                            continue
                            
                        # Calculate Manhattan distance between these cells
                        dist = abs(x1 - x2) + abs(y1 - y2)
                        
                        # Calculate number of ways to place remaining k-2 pieces
                        remaining_cells = m * n - 2
                        ways = nCr(remaining_cells, k-2)
                        
                        # Add to total
                        total = (total + dist * ways) % MOD
        
        # Each arrangement is counted twice (once for each ordering of the pair)
        total = (total * nCr(k, 2)) % MOD
        
        return total