from collections import Counter
class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        if m > n: return 0
        cnt, target = Counter(word2), Counter()
        for i, ch in enumerate(word1):
            cnt[ch] += 1
            if i >= m:
                cnt[word1[i-m]] -= 1
            if i >= m-1 and cnt == target+cnt:
                target = cnt.copy()
                cnt = Counter(word1[i-m+1:i+1])
                cnt[ch] += 1
        ans = n-m+1 if target == cnt else n-m
        subcnt = Counter(word1[:m])
        if not m or subcnt == cnt:
            ans += 1
        for st in range(1, n-m+1):
            subcnt[word1[st-1]] -= 1
            subcnt[word1[st+m-1]] += 1
            if subcnt == cnt:
                ans += 1
        return ans