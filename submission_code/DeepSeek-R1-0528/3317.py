from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        cnt = Counter()
        for word in words:
            for ch in word:
                cnt[ch] += 1
        total_pairs = sum(v // 2 for v in cnt.values())
        pair_required = [len(word) // 2 for word in words]
        pair_required.sort()
        ans = 0
        current_used = 0
        for req in pair_required:
            if current_used + req <= total_pairs:
                current_used += req
                ans += 1
            else:
                break
        return ans