from typing import List
from collections import deque

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        maxv = max(nums)
        # We'll consider target values t in [1..maxv+1]
        M = maxv + 1
        # frequency of original values
        freq = [0] * (M + 1)
        for x in nums:
            freq[x] += 1
        # A[t] = freq[t] + freq[t-1], t=1..M
        A = [0] * (M + 1)
        for t in range(1, M + 1):
            A[t] = freq[t] + (freq[t-1] if t-1 >= 0 else 0)
        # nextZero[t] = smallest idx >= t where A[idx]==0, or M+1 if none
        nextZero = [0] * (M + 2)
        nz = M + 1
        for t in range(M, 0, -1):
            if A[t] == 0:
                nz = t
            nextZero[t] = nz
        nextZero[M+1] = M + 1
        # prefix sums FPS[i] = sum freq[1..i]
        FPS = [0] * (M + 1)
        s = 0
        for i in range(1, M + 1):
            s += freq[i]
            FPS[i] = s
        # G[i] = FPS[i] - i, i=0..M
        G = [0] * (M + 1)
        # note G[0] = FPS[0] - 0 = 0
        for i in range(1, M + 1):
            G[i] = FPS[i] - i

        # Check if there exists a window of length L feasible
        def check(L: int) -> bool:
            # sliding window min on G over window size L
            # we need min G[k..k+L-1] for k=1..M-L+1
            deq = deque()
            minG = [0] * (M + 2)  # only entries 1..M-L+1 used
            # initialize deque for first L-1 elements
            for i in range(1, L):
                while deq and G[deq[-1]] >= G[i]:
                    deq.pop()
                deq.append(i)
            # process windows
            end = M - L + 1
            for k in range(1, end + 1):
                # include i = k+L-1
                idx = k + L - 1
                while deq and G[deq[-1]] >= G[idx]:
                    deq.pop()
                deq.append(idx)
                # evict k-1
                if deq[0] < k:
                    deq.popleft()
                minG[k] = G[deq[0]]
            # now test each start k
            for k in range(1, end + 1):
                # must have no zero in [k..k+L-1]
                r = k + L - 1
                if nextZero[k] <= r:
                    continue
                # available total = sum freq[k-1..r]
                if k >= 2:
                    tot = FPS[r] - FPS[k-2]
                else:
                    tot = FPS[r]
                if tot < L:
                    continue
                # prefix constraint: minG[k] >= base
                if k >= 2:
                    base = G[k-2] - 1
                else:
                    base = 0
                if minG[k] >= base:
                    return True
            return False

        # binary search on answer L in [1..n]
        lo, hi = 1, n
        ans = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans