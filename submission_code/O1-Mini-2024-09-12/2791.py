from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set()
        current = 1
        received.add(current)
        turn = 1
        while True:
            next_friend = (current + turn * k - 1) % n + 1
            if next_friend in received:
                break
            received.add(next_friend)
            current = next_friend
            turn += 1
        losers = [friend for friend in range(1, n + 1) if friend not in received]
        return losers