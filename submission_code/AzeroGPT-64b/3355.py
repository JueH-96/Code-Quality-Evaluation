class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n, alice = len(possible), 0
        bob = sum(possible)
        for i in range(n):
            if possible[i]:
                alice += 1
                bob -= 1
                if alice > bob:
                    return i + 1
        return -1 if n > 2 else 2