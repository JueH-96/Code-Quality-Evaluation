class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        changes = 0
        freq = [0]*26
        for _ in range(k):
            freq2 = [0]*26
            for i in range(n//k*k, min(n, n//k*k+k)):
                j = ord(s[i]) - ord('a')
                freq2[j] += 1
            changes += sum((freq2[j] - freq[j])**2 for j in range(26))
            freq = freq2
        return changes