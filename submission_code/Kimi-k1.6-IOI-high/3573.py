from collections import Counter
import bisect

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        need = Counter(word2)
        m = len(word2)
        n = len(word1)
        
        # Check if word1 has enough characters to satisfy word2's needs
        total_counts = Counter(word1)
        for c in need:
            if total_counts.get(c, 0) < need[c]:
                return 0
        
        # Precompute prefix sums for each character in need
        prefix = {c: [0] * (n + 1) for c in need}
        for c in need:
            count = 0
            for i in range(n):
                if word1[i] == c:
                    count += 1
                prefix[c][i+1] = count
        
        ans = 0
        # Iterate over all possible end indices of the substring in word1
        for right in range(m - 1, n):
            valid = True
            K = {}
            # Calculate K_c for each character in need
            for c in need:
                current = prefix[c][right + 1]
                required = need[c]
                K[c] = current - required
                if K[c] < 0:
                    valid = False
                    break
            if not valid:
                continue
            
            # Find the upper bounds for each character's prefix sum
            upper_c = []
            for c in need:
                # Find the largest index i where prefix[c][i] <= K[c]
                idx = bisect.bisect_right(prefix[c], K[c]) - 1
                upper_c.append(idx)
            
            min_upper = min(upper_c)
            max_i = min(min_upper, right - m + 1)
            if max_i >= 0:
                ans += (max_i + 1)
        
        return ans