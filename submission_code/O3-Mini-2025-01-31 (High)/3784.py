from collections import defaultdict
from typing import List

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        # If removing any word causes less than k words to remain, answer is 0.
        if n - 1 < k:
            return [0] * n

        # Precompute prefix counts: For each prefix length L, store a dictionary mapping prefix->count.
        prefix_counts = defaultdict(lambda: defaultdict(int))
        max_len = 0
        for word in words:
            Lw = len(word)
            if Lw > max_len:
                max_len = Lw
            prefix = ""
            for i in range(len(word)):
                prefix += word[i]
                prefix_counts[i+1][prefix] += 1

        # For each prefix length L (1-indexed up to max_len), compute:
        #  - global_max[L]: the highest frequency among all prefixes of that length.
        #  - best_prefix[L]: a prefix which attains that frequency.
        #  - global_count[L]: how many prefixes get this max.
        #  - global_second[L]: second highest frequency among prefixes of that length.
        global_max = [0] * (max_len + 1)
        best_prefix = [None] * (max_len + 1)
        global_count = [0] * (max_len + 1)
        global_second = [0] * (max_len + 1)
        
        for L in range(1, max_len + 1):
            if L not in prefix_counts:
                global_max[L] = 0
                best_prefix[L] = None
                global_count[L] = 0
                global_second[L] = 0
                continue
            best = 0
            count_best = 0
            second = 0
            bp = None
            for pref, freq in prefix_counts[L].items():
                if freq > best:
                    second = best
                    best = freq
                    count_best = 1
                    bp = pref
                elif freq == best:
                    count_best += 1
                elif freq > second:
                    second = freq
            global_max[L] = best
            best_prefix[L] = bp
            global_count[L] = count_best
            global_second[L] = second

        # Precompute the global answer (if no word were removed) for each L:
        # For each L, if global_max[L] >= k then that prefix length works.
        global_ans = 0
        for L in range(1, max_len + 1):
            if global_max[L] >= k:
                global_ans = L  # We update L in increasing order to finally get the maximum L

        # Now process each removal.
        # When removing a word, for lengths L greater than len(word_removed) the global frequency is unaffected.
        # For L ≤ len(word_removed), if the removed word’s prefix equals best_prefix[L] and it is uniquely best,
        # then the new maximum for that level becomes max(global_max[L]-1, global_second[L]).
        ans = [0] * n
        for i, word in enumerate(words):
            rlen = len(word)
            # Candidate from unaffected region (L > rlen): if global answer is > rlen, then it is unaffected.
            unaffected_candidate = 0
            if global_ans > rlen:
                unaffected_candidate = global_ans
            
            # Now check L from 1 to min(rlen, max_len).
            affected_candidate = 0
            upTo = rlen if rlen <= max_len else max_len
            cur_pref = ""
            for L in range(1, upTo + 1):
                cur_pref += word[L-1]
                # Determine the new frequency for this level after removal.
                if best_prefix[L] is not None and cur_pref == best_prefix[L] and global_count[L] == 1:
                    new_val = global_max[L] - 1
                    if global_second[L] > new_val:
                        new_val = global_second[L]
                else:
                    new_val = global_max[L]
                if new_val >= k:
                    affected_candidate = L  # We want the maximum L in the affected range that still works.
            
            ans[i] = max(unaffected_candidate, affected_candidate)
        return ans

# For local testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    words1 = ["jump","run","run","jump","run"]
    k1 = 2
    print(sol.longestCommonPrefix(words1, k1))  # Expected output: [3,4,4,3,4]
    
    # Example 2
    words2 = ["dog","racer","car"]
    k2 = 2
    print(sol.longestCommonPrefix(words2, k2))  # Expected output: [0,0,0]