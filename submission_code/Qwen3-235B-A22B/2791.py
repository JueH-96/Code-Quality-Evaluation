from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * (n + 1)
        current = 1
        visited[current] = True
        turn = 1
        
        while True:
            m = turn * k
            new_pos = (current - 1 + m) % n + 1
            if visited[new_pos]:
                break
            visited[new_pos] = True
            current = new_pos
            turn += 1
        
        losers = [i for i in range(1, n + 1) if not visited[i]]
        return losers