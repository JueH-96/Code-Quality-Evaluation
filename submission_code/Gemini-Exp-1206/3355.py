class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        for i in range(1, n):
            alice_score = 0
            bob_score = 0
            for j in range(i):
                if possible[j] == 1:
                    alice_score += 1
                else:
                    alice_score -= 1
            for j in range(i, n):
                if possible[j] == 1:
                    bob_score += 1
                else:
                    bob_score -= 1
            if alice_score > bob_score:
                return i
        return -1