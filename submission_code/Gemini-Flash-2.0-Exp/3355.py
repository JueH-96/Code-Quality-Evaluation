class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        for i in range(1, n):
            alice_score = sum(1 if possible[j] == 1 else -1 for j in range(i))
            bob_score = sum(1 if possible[j] == 1 else -1 for j in range(i, n))
            if alice_score > bob_score:
                return i
        return -1