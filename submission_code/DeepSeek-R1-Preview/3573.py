from collections import Counter
import bisect

class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        target_counts = Counter(word2)
        required_length = len(word2)
        n = len(word1)
        
        if required_length == 0:
            return 0
        if n < required_length:
            return 0
        
        word1_counts = Counter(word1)
        for c in target_counts:
            if word1_counts.get(c, 0) < target_counts[c]:
                return 0
        
        # Precompute prefix sums for each character in target_counts
        prefix = {}
        for c in target_counts:
            prefix[c] = [0] * (n + 1)
            current = 0
            for i in range(n):
                if word1[i] == c:
                    current += 1
                prefix[c][i + 1] = current
        
        total = 0
        
        for i in range(n):
            j_candidates = []
            valid = True
            for c in target_counts:
                required = target_counts[c]
                current_i_prev = prefix[c][i]
                required_total = current_i_prev + required
                low = i + 1
                high = n
                res = bisect.bisect_left(prefix[c], required_total, lo=low, hi=high + 1)
                if res > high:
                    valid = False
                    break
                j_candidate = res - 1
                j_candidates.append(j_candidate)
            if not valid:
                continue
            j_counts = max(j_candidates)
            j_min = max(j_counts, i + required_length - 1)
            if j_min >= n:
                continue
            total += (n - j_min)
        return total