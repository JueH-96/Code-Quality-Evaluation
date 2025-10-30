from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = [False] * (n + 1)
        current = 1
        visited[current] = True
        i = 1
        
        while True:
            step = i * k
            # compute next friend in 1-based index
            current = (current - 1 + step) % n + 1
            if visited[current]:
                break
            visited[current] = True
            i += 1
        
        # collect those who never received the ball
        losers = [friend for friend in range(1, n + 1) if not visited[friend]]
        return losers