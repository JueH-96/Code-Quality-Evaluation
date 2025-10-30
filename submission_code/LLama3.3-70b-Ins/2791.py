from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * n
        current = 0
        step = 1
        while not visited[current]:
            visited[current] = True
            current = (current + step * k) % n
            step += 1
        losers = [i + 1 for i, v in enumerate(visited) if not v]
        return losers