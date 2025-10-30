from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        # visited[i] will be True if friend i has ever received the ball
        visited = [False] * (n + 1)
        
        # start with friend 1
        cur = 1
        visited[cur] = True
        
        # on the i-th pass, we move i*k steps
        step = 1
        while True:
            # compute next friend in 1-based indexing
            cur = (cur - 1 + step * k) % n + 1
            # if this friend already had the ball, game ends
            if visited[cur]:
                break
            visited[cur] = True
            step += 1
        
        # losers are those who never received the ball
        return [i for i in range(1, n + 1) if not visited[i]]