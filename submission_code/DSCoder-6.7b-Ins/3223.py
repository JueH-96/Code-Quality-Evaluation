class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        from collections import Counter
        n = len(word)
        count = Counter()
        res = 0
        j = 0
        for i in range(n):
            while j < n and len(count) < k:
                count[word[j]] += 1
                j += 1
            if len(count) == k:
                min_count = min(count.values())
                res += min_count
            count[word[i]] -= 1
            if count[word[i]] == 0:
                del count[word[i]]
        return res