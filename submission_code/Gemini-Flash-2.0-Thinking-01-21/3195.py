class Solution:
    def minimumSteps(self, s: str) -> int:
        n = len(s)
        inversions = 0
        for i in range(n):
            if s[i] == '1':
                for j in range(i + 1, n):
                    if s[j] == '0':
                        inversions += 1
        return inversions