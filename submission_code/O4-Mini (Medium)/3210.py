from collections import defaultdict

class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(s)
        # prefix_v[i] = number of vowels in s[:i]
        # prefix_diff[i] = (vowels - consonants) in s[:i]
        prefix_v = [0] * (n + 1)
        prefix_diff = [0] * (n + 1)
        
        for i, ch in enumerate(s):
            is_v = 1 if ch in vowels else 0
            prefix_v[i + 1] = prefix_v[i] + is_v
            prefix_diff[i + 1] = prefix_diff[i] + (1 if is_v else -1)
        
        # Group prefix‐indices by their diff value.
        groups = defaultdict(list)
        for idx, d in enumerate(prefix_diff):
            groups[d].append(idx)
        
        ans = 0
        # For each group of equal diffs, every pair (i<j) gives a substring
        # with vowels == consonants. Check the product condition.
        for diff_val, L in groups.items():
            m = len(L)
            for a in range(m):
                i = L[a]
                for b in range(a + 1, m):
                    j = L[b]
                    # Number of vowels in s[i:j]
                    v = prefix_v[j] - prefix_v[i]
                    # Since vowels == consonants, c == v and we need v*v % k == 0
                    if (v * v) % k == 0:
                        ans += 1
        
        return ans