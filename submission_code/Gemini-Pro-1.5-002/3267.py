class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        ans = -1
        for length in range(1, n // 3 + 1):
            counts = {}
            for i in range(n - length + 1):
                substring = s[i:i + length]
                if all(c == substring[0] for c in substring):
                    counts[substring] = counts.get(substring, 0) + 1
            for count in counts.values():
                if count >= 3:
                    ans = max(ans, length)
        return ans