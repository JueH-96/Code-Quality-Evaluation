from typing import List, Dict
from collections import defaultdict

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        # if after removal there are fewer than k words then answer is 0 for that removal.
        if n - 1 < k:
            return [0] * n

        # 1. Precompute global prefix counts.
        # Because the sum of lengths is <= 1e5, we can iterate through each word.
        prefix_count: Dict[str, int] = defaultdict(int)
        max_len = 0
        for word in words:
            max_len = max(max_len, len(word))
            # For each prefix in the word
            prefix = ""
            for ch in word:
                prefix += ch
                prefix_count[prefix] += 1

        # 2. For each possible prefix length L = 1,...,max_len,
        # count how many distinct prefixes of length L have count >= k.
        distinct_valid = [0] * (max_len + 1)  # 1-indexed: distinct_valid[L]
        for p, ct in prefix_count.items():
            L = len(p)
            if L <= max_len and ct >= k:
                distinct_valid[L] += 1
        # Note: for a given L, if distinct_valid[L] is 0 then no prefix of that length is valid,
        # and if positive, then at least one candidate exists globally.
        
        # 3. For each unique word s we want to compute:
        #    ans(s) = maximum L in [1, max_len] such that:
        #         if L <= len(s):
        #             (distinct_valid[L]>=2) OR (distinct_valid[L]==1 and prefix_count[s[:L]]>=k+1)
        #         if L > len(s):
        #             (distinct_valid[L]>=1)
        # We can binary search for the maximum L with condition True because the feasibility is monotonic.
        # Explanation on monotonicity:
        #   If some length L is feasible, then all smaller lengths are feasible. (A common prefix of length L implies a common prefix of all smaller lengths.)
        # So we can binary search on L for each unique s.
        
        # Precompute answers for each unique word.
        # We'll search L in [1, max_len]. If no L satisfies, answer is 0.
        ans_for_word = {}
        
        for s in set(words):
            lo, hi = 1, max_len
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                # check condition for length mid
                if mid <= len(s):
                    # s contributes to the candidate prefix s[:mid].
                    # Two cases: either there is at least a second valid prefix OR
                    # s[:mid] is the unique valid candidate but its count is at least k+1.
                    if distinct_valid[mid] >= 2 or (distinct_valid[mid] == 1 and prefix_count[s[:mid]] >= k + 1):
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                else:
                    # mid > len(s): s does not contribute.
                    # So condition is: exists any valid prefix of length mid.
                    if distinct_valid[mid] >= 1:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
            ans_for_word[s] = best

        # 4. Build output for each index using the precomputed value.
        res = [ans_for_word[word] for word in words]
        return res