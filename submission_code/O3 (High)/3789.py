from typing import List, DefaultDict
from collections import defaultdict


class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        m = len(conflictingPairs)
        # bucket[b] – list of (a , pairId) whose right end is b
        bucket: DefaultDict[int, List[tuple]] = defaultdict(list)
        for idx, (x, y) in enumerate(conflictingPairs):
            a, b = (x, y) if x < y else (y, x)
            bucket[b].append((a, idx))

        gain = [0] * m          # gain of every pair
        baseline = 0            # sub-arrays with all pairs present

        maxA = 0                # current maximal  a
        cntMax = 0              # how many pairs reach maxA
        maxId = -1              # id of the unique maximal pair (only if cntMax == 1)
        secondA = 0             # second largest  a

        for r in range(1, n + 1):
            # insert all pairs whose right end == r
            for a, pid in bucket.get(r, ()):
                if a > maxA:
                    secondA = maxA
                    maxA = a
                    cntMax = 1
                    maxId = pid
                elif a == maxA:
                    cntMax += 1
                    maxId = -1        # now ambiguous
                else:  # a < maxA
                    if a > secondA:
                        secondA = a

            # smallest allowed left border
            left = maxA + 1
            # valid sub-arrays ending at r
            baseline += r - left + 1

            # if the current maximal a is unique – add its gain
            if cntMax == 1 and maxId != -1:
                diff = maxA - secondA  # secondA may be 0
                gain[maxId] += diff

        return baseline + max(gain, default=0)