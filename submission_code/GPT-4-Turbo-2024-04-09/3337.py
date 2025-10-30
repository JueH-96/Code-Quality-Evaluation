class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        positions = [i for i in range(len(s)) if s[i] == c]
        count = 0
        for i in range(len(positions)):
            for j in range(i, len(positions)):
                count += positions[j] - positions[i] + 1
        return count