class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        prefix = [0] * (n+1)
        suffix = [0] * (n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + ord(s[i])
            suffix[i+1] = suffix[i] + ord('z') - ord(s[i])
        min_substrings = float('inf')
        for i in range(n):
            for j in range(i+1, n+1):
                total = prefix[j] - prefix[i] + suffix[i] - suffix[j]
                min_substrings = min(min_substrings, total)
        return min_substrings