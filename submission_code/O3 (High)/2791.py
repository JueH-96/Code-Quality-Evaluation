from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set([1])          # friends that have already received the ball
        current = 1                 # friend that is currently holding the ball
        step = 1                    # i-th turn (distance multiplier)

        while True:
            # friend that will receive the ball this turn
            nxt = ((current - 1 + step * k) % n) + 1   # keep numbering 1 … n

            if nxt in visited:       # someone gets the ball twice → game ends
                break

            visited.add(nxt)
            current = nxt
            step += 1

        # friends that never touched the ball are the losers
        return [friend for friend in range(1, n + 1) if friend not in visited]