from bisect import bisect_right
from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        # Quick check
        if m > n:
            return []

        # Build lists of positions for each character in word1
        letter_positions1 = [[] for _ in range(26)]
        for i, ch in enumerate(word1):
            letter_positions1[ord(ch) - 97].append(i)

        # Build fwd array and posL for greedy exact-match prefix
        fwd = [0] * n
        posL = [0] * m
        j = 0
        for i, ch in enumerate(word1):
            if j < m and ch == word2[j]:
                posL[j] = i
                j += 1
            fwd[i] = j
        # If we matched the whole word2 as a subsequence, record exact_seq
        if j == m:
            exact_seq = posL  # Already length m
        else:
            exact_seq = None

        # Build bwd array for greedy exact-match suffix (from the end)
        bwd = [0] * n
        j = 0
        for i in range(n - 1, -1, -1):
            if j < m and word1[i] == word2[m - 1 - j]:
                j += 1
            bwd[i] = j

        # Build prev_dif so that prev_dif[r] = largest idx < r where word2[idx] != word2[r]
        prev_dif = [-1] * m
        for i in range(1, m):
            if word2[i] != word2[i - 1]:
                prev_dif[i] = i - 1
            else:
                prev_dif[i] = prev_dif[i - 1]

        # Scan for the earliest position p in word1 where we can put our one mismatch
        mis_p = -1
        mis_t = -1
        for p in range(n):
            # how many of word2's prefix we matched before index p
            prefix_count = fwd[p - 1] if p > 0 else 0
            # how many of word2's suffix we can match after index p
            suffix_count = bwd[p + 1] if p + 1 < n else 0
            # we need prefix_count + suffix_count >= m-1 to fit the rest
            if prefix_count + suffix_count < m - 1:
                continue

            # valid mismatch index t must lie in [L..R]
            L = m - 1 - suffix_count
            if L < 0:
                L = 0
            R = prefix_count
            if R >= m:
                R = m - 1
            if L > R:
                continue

            c = word1[p]
            # if word2[R] != c, we can mismatch at t = R
            if word2[R] != c:
                t = R
            else:
                # else try the previous position where word2[...] != c
                t0 = prev_dif[R]
                if t0 < L:
                    continue
                t = t0

            # found a valid place for our one mismatch
            mis_p, mis_t = p, t
            break

        # Build the mismatch sequence if found
        mismatch_seq = None
        if mis_p != -1:
            seq = [0] * m
            # exact-match prefix up to t-1
            for idx in range(mis_t):
                seq[idx] = posL[idx]
            # the mismatch position
            seq[mis_t] = mis_p
            # fill the suffix exactly
            cur_idx = mis_p
            for idx in range(mis_t + 1, m):
                c = word2[idx]
                lst = letter_positions1[ord(c) - 97]
                # binary-search for the next position > cur_idx
                jpos = bisect_right(lst, cur_idx)
                # it must exist because of our suffix_count check
                seq[idx] = lst[jpos]
                cur_idx = seq[idx]
            mismatch_seq = seq

        # Decide between exact_seq and mismatch_seq
        if exact_seq is None:
            return mismatch_seq or []
        if mismatch_seq is None:
            # return a copy if you need to avoid side-effects,
            # but posL itself is safe to return
            return exact_seq

        # both exist: lex-compare
        for a, b in zip(exact_seq, mismatch_seq):
            if a < b:
                return exact_seq
            if a > b:
                return mismatch_seq
        # they are equal
        return exact_seq