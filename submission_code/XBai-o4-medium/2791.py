from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        current = 1
        visited.add(current)
        i = 1  # turn counter
        
        while True:
            steps = i * k
            next_pos = (current - 1 + steps) % n + 1
            if next_pos in visited:
                break
            visited.add(next_pos)
            current = next_pos
            i += 1
        
        losers = []
        for j in range(1, n + 1):
            if j not in visited:
                losers.append(j)
        return losers