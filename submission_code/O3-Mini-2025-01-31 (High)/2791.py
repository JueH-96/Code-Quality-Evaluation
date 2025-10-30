from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        visited = set()
        current = 0  # using 0-index; friend 1 is index 0
        turn = 1
        visited.add(current)

        while True:
            next_friend = (current + turn * k) % n
            if next_friend in visited:
                break
            visited.add(next_friend)
            current = next_friend
            turn += 1

        # Determine losers: friends who never received the ball (convert index to friend number)
        losers = [i + 1 for i in range(n) if i not in visited]
        return losers