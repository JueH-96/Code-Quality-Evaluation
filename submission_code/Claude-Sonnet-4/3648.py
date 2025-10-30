class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        
        # The key insight is that we can solve this step by step
        # At each step, we track possible positions of all three children
        
        from functools import lru_cache
        
        @lru_cache(None)
        def solve(step, r1, c1, r2, c2, r3, c3):
            if step == n - 1:
                # All must reach (n-1, n-1)
                if r1 == n-1 and c1 == n-1 and r2 == n-1 and c2 == n-1 and r3 == n-1 and c3 == n-1:
                    return fruits[n-1][n-1]
                return 0
            
            # Collect fruits from current positions (avoid double counting)
            positions = set([(r1, c1), (r2, c2), (r3, c3)])
            current_score = sum(fruits[r][c] for r, c in positions)
            
            max_next = 0
            
            # Generate all valid moves for each child
            # Child 1: (i+1,j+1), (i+1,j), (i,j+1)
            for dr1, dc1 in [(1,1), (1,0), (0,1)]:
                nr1, nc1 = r1 + dr1, c1 + dc1
                if not (0 <= nr1 < n and 0 <= nc1 < n):
                    continue
                    
                # Child 2: (i+1,j-1), (i+1,j), (i+1,j+1)
                for dr2, dc2 in [(1,-1), (1,0), (1,1)]:
                    nr2, nc2 = r2 + dr2, c2 + dc2
                    if not (0 <= nr2 < n and 0 <= nc2 < n):
                        continue
                        
                    # Child 3: (i-1,j+1), (i,j+1), (i+1,j+1)
                    for dr3, dc3 in [(-1,1), (0,1), (1,1)]:
                        nr3, nc3 = r3 + dr3, c3 + dc3
                        if not (0 <= nr3 < n and 0 <= nc3 < n):
                            continue
                            
                        next_score = solve(step + 1, nr1, nc1, nr2, nc2, nr3, nc3)
                        max_next = max(max_next, next_score)
            
            return current_score + max_next
        
        return solve(0, 0, 0, 0, n-1, n-1, 0)