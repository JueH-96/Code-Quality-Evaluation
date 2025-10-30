class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        cnt = [0]*26
        for c in s:
            cnt[ord(c)-ord('a')] += 1
        min_substring_changes = [0]*26
        for i in range(26):
            if cnt[i] > 0:
                min_substring_changes[i] = n - cnt[i]
        changes = 0
        for c in s:
            changes += min_substring_changes[ord(c)-ord('a')]
        return changes