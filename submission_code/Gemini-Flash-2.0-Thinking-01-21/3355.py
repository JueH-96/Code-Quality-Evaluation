class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        for k in range(1, n):
            alice_score = 0
            for i in range(k):
                if possible[i] == 1:
                    alice_score += 1
                else:
                    alice_score -= 1
            bob_score = 0
            for i in range(k, n):
                if possible[i] == 1:
                    bob_score += 1
                else:
                    bob_score -= 1
            if alice_score > bob_score:
                return k
        return -1