from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * n
        current_position = 0
        turn = 1
        
        while not visited[current_position]:
            visited[current_position] = True
            current_position = (current_position + turn * k) % n
            turn += 1
        
        losers = [i + 1 for i in range(n) if not visited[i]]
        return sorted(losers)