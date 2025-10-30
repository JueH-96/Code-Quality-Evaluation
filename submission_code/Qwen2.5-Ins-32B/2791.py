from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        current = 0
        i = 1
        while current not in visited:
            visited.add(current)
            current = (current + i * k) % n
            i += 1
        
        losers = [friend + 1 for friend in range(n) if friend not in visited]
        return losers