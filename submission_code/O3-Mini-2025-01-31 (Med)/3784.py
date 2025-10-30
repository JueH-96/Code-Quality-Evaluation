from typing import List
import collections

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        # When removing one word, if remaining words < k then answer is 0.
        if n - 1 < k:
            return [0] * n

        # We will “count” how many words have a given prefix.
        # Because the total sum of word lengths is at most 10^5, if we loop over every prefix of every word the cost is acceptable.
        # Let L_max be the length of the longest word.
        L_max = 0
        for w in words:
            if len(w) > L_max:
                L_max = len(w)

        # Build an array (indexed by prefix length L = 0...L_max) holding a Counter that maps each prefix (of length L)
        # to its frequency. We only need lengths 1..L_max.
        counts_by_length = [None] * (L_max + 1)
        for L in range(L_max + 1):
            counts_by_length[L] = collections.Counter()
            
        for w in words:
            # For each w, update counts for every possible prefix length from 1 to len(w)
            for L in range(1, len(w) + 1):
                pref = w[:L]
                counts_by_length[L][pref] += 1

        # For each possible prefix length L (1 to L_max), we determine:
        #   best_count[L]   : the maximum frequency among any prefix of length L.
        #   best_prefix[L]  : a prefix that attains best_count[L].
        #   second_count[L] : the second highest frequency (if exists, else 0).
        best_count = [0] * (L_max + 1)
        second_count = [0] * (L_max + 1)
        best_prefix = [""] * (L_max + 1)
        for L in range(1, L_max + 1):
            best = 0
            second = 0
            bestPref = ""
            for pref, cnt in counts_by_length[L].items():
                if cnt > best:
                    second = best
                    best = cnt
                    bestPref = pref
                elif cnt > second:
                    second = cnt
            best_count[L] = best
            second_count[L] = second
            best_prefix[L] = bestPref

        # In the original (global) collection of words the property “there is some prefix of length L 
        # that appears in at least k words” holds if best_count[L] >= k.
        globalEligible = [False] * (L_max + 1)
        for L in range(1, L_max + 1):
            if best_count[L] >= k:
                globalEligible[L] = True

        # For L > some length r, removal does not affect since the removed word does not even contribute to those prefixes.
        # Precompute a "suffix" array: for each L from 1 to L_max+1, let globalSuffix[L] be the maximum L' in [L, L_max]
        # for which globalEligible[L'] is True. (If-none then 0.)
        globalSuffix = [0] * (L_max + 2)
        globalSuffix[L_max + 1] = 0
        for L in range(L_max, 0, -1):
            if globalEligible[L]:
                globalSuffix[L] = L
            else:
                globalSuffix[L] = globalSuffix[L + 1]

        # For each removal index i in words, we want the answer:
        # After removing the i-th word r, we are allowed to choose any k words from the remainder.
        # Consider any candidate prefix p with length L.
        # The frequency p will have in the remainder is:
        #   • if r has p as a prefix then it contributes 1 less, i.e. count = global_count(p) - 1,
        #   • otherwise it remains global_count(p).
        # Hence, for each L, the maximum achievable count in the remainder is:
        #   if r’s prefix of length L (namely r[:L]) is the one that obtains global best_count[L],
        #       effective = max( best_count[L] - 1, second_count[L] )
        #   else:
        #       effective = best_count[L]
        # (For L greater than len(r), r doesn’t contribute so effective = best_count[L].)
        # Then answer[i] is the maximum L (1 <= L <= L_max) for which effective >= k.
        # Notice that even though the “existence” property isn’t monotonic in L in general,
        # the sum of lengths of all words is small so we can iterate over L from 1 to at most len(r) for each removed word.
        
        res = [0] * n
        for i, r in enumerate(words):
            bestL = 0
            r_len = len(r)
            # Check lengths L = 1 to min(len(r), L_max):
            for L in range(1, min(r_len, L_max) + 1):
                # r must appear in counts_by_length[L], so:
                cnt_r = counts_by_length[L][r[:L]]
                # If r's prefix exactly equals the best prefix at length L:
                if r[:L] == best_prefix[L]:
                    # After removal, if we choose that same prefix p (which r helped count),
                    # its new count would be best_count[L]-1.
                    # But it may be that some other prefix p' (with frequency second_count[L]) is available unchanged.
                    effective = best_count[L] - 1 if (best_count[L] - 1) >= second_count[L] else second_count[L]
                else:
                    effective = best_count[L]
                if effective >= k and L > bestL:
                    bestL = L
            # For L > r_len (or if r_len < L_max), the removed word did not contribute so effective = best_count[L].
            candidate2 = globalSuffix[r_len + 1] if r_len < L_max else 0
            res[i] = bestL if bestL > candidate2 else candidate2
        return res


# --------------------------
# For local testing

if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    words1 = ["jump","run","run","jump","run"]
    k1 = 2
    # Expected output: [3, 4, 4, 3, 4]
    print(sol.longestCommonPrefix(words1, k1))
    
    # Example 2:
    words2 = ["dog","racer","car"]
    k2 = 2
    # Expected output: [0, 0, 0]
    print(sol.longestCommonPrefix(words2, k2))
    
    # Additional simple test.
    words3 = ["a", "ab", "abc"]
    k3 = 2
    print(sol.longestCommonPrefix(words3, k3))
    
    # Test with k equal to number of words.
    words4 = ["apple", "apply", "apt", "ape"]
    k4 = 4
    print(sol.longestCommonPrefix(words4, k4))
    
    # Test where removal leaves fewer than k words.
    words5 = ["one", "two"]
    k5 = 2
    # Expected, because removal leaves 1 word which is < k, so answer should be [0, 0].
    print(sol.longestCommonPrefix(words5, k5))