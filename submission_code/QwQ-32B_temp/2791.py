from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [0] * (n + 1)
        visited[1] = 1
        current = 1
        turn = 1
        
        while True:
            steps = turn * k
            next_pos = (current - 1 + steps) % n + 1
            if visited[next_pos] > 0:
                break
            visited[next_pos] += 1
            current = next_pos
            turn += 1
        
        losers = []
        for i in range(1, n + 1):
            if visited[i] == 0:
                losers.append(i)
        
        return losers