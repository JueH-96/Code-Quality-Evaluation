from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set([1])
        current = 1
        step = 1
        
        while True:
            # compute next friend in 1-based indexing
            next_friend = (current - 1 + step * k) % n + 1
            if next_friend in visited:
                break
            visited.add(next_friend)
            current = next_friend
            step += 1
        
        # losers are those who never received the ball
        return [friend for friend in range(1, n+1) if friend not in visited]