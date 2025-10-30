class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def distance(x1, y1, x2, y2):
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            return max((max(dx, dy) + 1) // 2, (dx + dy + 2) // 3)

        n = len(positions)
        distances = [distance(kx, ky, x, y) for x, y in positions]
        
        @functools.lru_cache(None)
        def dfs(mask):
            if mask == (1 << n) - 1:
                return 0
            
            alice_best = -float('inf')
            for i in range(n):
                if mask & (1 << i) == 0:
                    bob_best = float('inf')
                    for j in range(n):
                        if i != j and mask & (1 << j) == 0:
                            bob_move = dfs(mask | (1 << i) | (1 << j))
                            bob_best = min(bob_best, bob_move)
                    
                    if bob_best == float('inf'):
                        alice_best = max(alice_best, distances[i])
                    else:
                        alice_best = max(alice_best, distances[i] + bob_best)
            
            return alice_best
        
        return dfs(0)