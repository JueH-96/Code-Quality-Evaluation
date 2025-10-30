from typing import List
import bisect

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        # build positions of each letter in word1
        pos = [[] for _ in range(26)]
        for i, ch in enumerate(word1):
            pos[ord(ch) - 97].append(i)
        # 1) try perfect match: build pref[j] = smallest index matching word2[0..j]
        pref = [0] * m
        last = -1
        fail = m  # first j where we fail
        for j in range(m):
            c = ord(word2[j]) - 97
            lst = pos[c]
            # find index > last
            k = bisect.bisect_right(lst, last)
            if k == len(lst):
                fail = j
                break
            idx = lst[k]
            pref[j] = idx
            last = idx
        # if perfect succeeded
        if fail == m:
            return pref
        k = fail  # mismatch position
        # 2) build suffPos[j]: latest possible starting positions to match word2[j..]
        suffPos = [0] * m
        nxt = n  # we search before nxt
        for j in range(m - 1, -1, -1):
            c = ord(word2[j]) - 97
            lst = pos[c]
            # find index < nxt
            idx = bisect.bisect_left(lst, nxt) - 1
            if idx < 0:
                # can't match suffix starting at j
                suffPos[j] = -1
            else:
                suffPos[j] = lst[idx]
            nxt = suffPos[j]
        # 3) determine scan interval for mismatch at k
        prev = pref[k-1] if k > 0 else -1
        bound = suffPos[k+1] if k+1 < m else n
        # scan for first i in (prev, bound) with word1[i] != word2[k]
        target = word2[k]
        i0 = -1
        # linear scan once
        for i in range(prev+1, bound):
            if word1[i] != target:
                i0 = i
                break
        if i0 == -1:
            return []
        # 4) build the full result: pref[0..k-1], then i0, then greedy match j>k
        res = []
        for j in range(k):
            res.append(pref[j])
        res.append(i0)
        last = i0
        # suffix from k+1..m-1
        for j in range(k+1, m):
            c = ord(word2[j]) - 97
            lst = pos[c]
            idx = bisect.bisect_right(lst, last)
            if idx == len(lst):
                return []
            p = lst[idx]
            res.append(p)
            last = p
        return res