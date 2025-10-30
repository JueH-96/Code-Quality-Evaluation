from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        alice_score = 0
        bob_score = sum(possible)

        for i in range(n):
            if possible[i] == 1:
                alice_score += 1
                bob_score -= 1

            if alice_score > bob_score:
                return i + 1

        return -1