from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * n
        current = 0  # Using 0-index for friend numbered 1
        turn = 1
        
        while not visited[current]:
            visited[current] = True
            # Determine next friend (adding turn*k steps, then mod n)
            current = (current + turn * k) % n
            turn += 1
        
        # Losers are friends (1-indexed) that never had the ball
        losers = [i + 1 for i, v in enumerate(visited) if not v]
        return losers