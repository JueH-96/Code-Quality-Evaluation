class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        def atLeastK(s, k):
            count = [0] * 26
            for char in s:
                count[ord(char) - ord('a')] += 1
            return all(c >= k for c in count if c > 0)

        n = len(s)
        result = 0

        for i in range(n):
            for j in range(i, n):
                if atLeastK(s[i:j+1], k):
                    result += 1

        return result