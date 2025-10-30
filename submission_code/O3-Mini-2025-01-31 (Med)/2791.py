from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * n
        current = 0  # starting from friend 1, index 0
        visited[current] = True
        step = 1
        
        while True:
            next_friend = (current + step * k) % n
            if visited[next_friend]:
                break
            visited[next_friend] = True
            current = next_friend
            step += 1
        
        losers = [i+1 for i in range(n) if not visited[i]]
        return losers