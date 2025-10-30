from typing import List

class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        received = set()
        received.add(1)
        current_holder = 1
        turn = 1
        while True:
            steps = turn * k
            next_holder = (current_holder + steps - 1) % n + 1
            if next_holder in received:
                break
            received.add(next_holder)
            current_holder = next_holder
            turn += 1
        losers = [friend for friend in range(1, n+1) if friend not in received]
        return losers