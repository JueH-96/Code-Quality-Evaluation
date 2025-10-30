class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        for k in range(1, n):
            alice_plays = possible[:k]
            bob_plays = possible[k:]

            alice_score = sum(alice_plays)
            bob_score = sum(bob_plays)

            if alice_score > bob_score:
                return k
        return -1