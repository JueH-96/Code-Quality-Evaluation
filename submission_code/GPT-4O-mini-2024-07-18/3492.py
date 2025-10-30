class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        from collections import defaultdict
        
        m, n = len(grid), len(grid[0])
        count = 0
        
        # Iterate over all possible top-left corners of submatrices
        for r1 in range(m):
            for c1 in range(n):
                if grid[r1][c1] != 'X':  # We need to start from grid[0][0]
                    continue
                
                # Frequency map for 'X' and 'Y'
                freq = defaultdict(int)
                
                # Iterate over all possible bottom-right corners of submatrices
                for r2 in range(r1, m):
                    for c2 in range(c1, n):
                        if grid[r2][c2] == 'X':
                            freq['X'] += 1
                        elif grid[r2][c2] == 'Y':
                            freq['Y'] += 1
                        
                        # Check if we have at least one 'X' and equal count of 'X' and 'Y'
                        if freq['X'] > 0 and freq['X'] == freq['Y']:
                            count += 1
        
        return count