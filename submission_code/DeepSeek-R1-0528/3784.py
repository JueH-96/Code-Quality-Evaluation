from collections import defaultdict

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        if n - 1 < k:
            return [0] * n
        
        freq = defaultdict(int)
        for word in words:
            for l in range(0, len(word) + 1):
                prefix = word[:l]
                freq[prefix] += 1
        
        cnt_len = defaultdict(int)
        for prefix, count in freq.items():
            if count >= k:
                l_val = len(prefix)
                cnt_len[l_val] += 1
        
        best_global = max(cnt_len.keys()) if cnt_len else 0
        
        ans = [0] * n
        for i in range(n):
            w = words[i]
            critical = set()
            for l in range(0, len(w) + 1):
                if l > best_global:
                    break
                prefix = w[:l]
                if freq[prefix] == k and l in cnt_len and cnt_len[l] == 1:
                    critical.add(l)
            candidate = best_global
            while candidate in critical:
                candidate -= 1
            ans[i] = candidate
        
        return ans