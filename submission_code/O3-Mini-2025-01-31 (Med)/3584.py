from typing import List
from bisect import bisect_left, bisect_right

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        INF = n+1  # a value larger than any valid index

        # Precompute for each letter a sorted list of positions
        posDict = {chr(c): [] for c in range(ord('a'), ord('z')+1)}
        for i, ch in enumerate(word1):
            posDict[ch].append(i)
        
        # helper: given a pattern (string) and starting index in word1, try to match pattern exactly.
        # Returns (True, sequence) if possible, else (False, None).
        def can_match(pattern: str, start: int):
            seq = []
            pos = start
            for ch in pattern:
                lst = posDict[ch]
                # find the leftmost index in lst that is >= pos using binary search
                j = bisect_left(lst, pos)
                if j == len(lst):
                    return False, None
                pos = lst[j] + 1
                seq.append(lst[j])
            return True, seq
        
        # First, try to match word2 exactly from index 0.
        ok, exact_seq = can_match(word2, 0)
        if ok:
            return exact_seq

        # Next, compute as many letters of word2 as we can match exactly from the beginning.
        prefix_seq = []
        cur = 0
        k_max = 0
        for i in range(m):
            lst = posDict[word2[i]]
            j = bisect_left(lst, cur)
            if j == len(lst):
                break
            pos_ch = lst[j]
            prefix_seq.append(pos_ch)
            cur = pos_ch + 1
            k_max = i + 1  # we matched i+1 letters exactly

        # Now try to force a mismatch at some position i (0-indexed in word2).
        # We try i from 0 to k_max (inclusive). Note: if k_max < m, then word2[k_max] could not be matched exactly.
        for i in range(0, k_max + 1):
            # For prefix exactly matching word2[0:i]:
            # When i==0, prefix is empty.
            # For i>=1, prefix_seq[0:i] is already determined.
            prefix_part = prefix_seq[:i]
            # For the mismatch at position i, choose the smallest possible index x.
            # If i==0, we can choose x = 0.
            # Otherwise, we choose x = prefix_seq[i-1] + 1.
            if i == 0:
                candidate = 0
            else:
                candidate = prefix_seq[i-1] + 1
            # Now, for the remaining part word2[i+1:], we require an exact match starting from candidate+1:
            ok2, suf_seq = can_match(word2[i+1:], candidate + 1)
            if ok2:
                candidate_seq = prefix_part + [candidate] + suf_seq
                return candidate_seq
        return []