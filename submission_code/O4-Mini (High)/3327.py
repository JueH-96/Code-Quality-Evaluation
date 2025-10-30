from typing import List
from collections import deque

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        p = maxChanges
        # Positions of original ones
        P = [i for i, v in enumerate(nums) if v == 1]
        m = len(P)
        # T = number of original ones we must pick at minimum
        T = k - p
        if T <= 0:
            # We can create all k by changes if desired (or mix),
            # best to use any nearby original ones at distance <=1.
            # For each center i, cost = neighbor_ones*1 + 2*(k - L)
            # where L = nums[i] + neighbor_ones.
            ans = 10**30
            for i in range(n):
                center = nums[i]
                nb = 0
                if i > 0 and nums[i-1] == 1:
                    nb += 1
                if i + 1 < n and nums[i+1] == 1:
                    nb += 1
                L = center + nb
                # cost = sum_d[L] + 2*(k - L), but sum_d[L] = nb*1
                cost = nb + 2 * (k - L)
                # or simplified: cost = 2*k - 2*center - nb
                # cost = 2*k - 2*center - nb
                # but we already compute direct cost
                if cost < ans:
                    ans = cost
            return ans

        # Now T = k-p > 0
        # If T >= 4, sliding-window on P for minimal sum of T nearest originals
        if T >= 4:
            # We need at least T original ones; guaranteed m >= T by constraints.
            S = [0] * (m + 1)
            for i in range(m):
                S[i+1] = S[i] + P[i]
            best = 10**30
            # slide a window of length T over P
            for l in range(0, m - T + 1):
                r = l + T - 1
                mid = l + (T // 2)
                med = P[mid]
                # sum of distances to med
                # left part: P[l..mid-1] to med
                left_count = mid - l
                left_sum = S[mid] - S[l]
                cost_left = med * left_count - left_sum
                # right part: P[mid+1..r] to med
                right_count = r - mid
                right_sum = S[r+1] - S[mid+1]
                cost_right = right_sum - med * right_count
                total = cost_left + cost_right
                if total < best:
                    best = total
            # plus the cost of creating p ones: 2 per created
            return best + 2 * p

        # Now T in {1,2,3}: build prev/next arrays for up to 3 nearest ones
        prev1 = [-1] * n
        prev2 = [-1] * n
        prev3 = [-1] * n
        dq = deque(maxlen=3)
        for i in range(n):
            if nums[i] == 1:
                dq.append(i)
            ln = len(dq)
            if ln >= 1:
                prev1[i] = dq[-1]
            if ln >= 2:
                prev2[i] = dq[-2]
            if ln >= 3:
                prev3[i] = dq[-3]

        nxt1 = [-1] * n
        nxt2 = [-1] * n
        nxt3 = [-1] * n
        dq = deque(maxlen=3)
        for i in range(n-1, -1, -1):
            if nums[i] == 1:
                dq.append(i)
            ln = len(dq)
            if ln >= 1:
                nxt1[i] = dq[-1]
            if ln >= 2:
                nxt2[i] = dq[-2]
            if ln >= 3:
                nxt3[i] = dq[-3]

        ans = 10**30
        # For each center i, compute cost
        for i in range(n):
            center = nums[i]
            nb = 0
            if i > 0 and nums[i-1] == 1:
                nb += 1
            if i + 1 < n and nums[i+1] == 1:
                nb += 1
            L = center + nb
            # t_opt = max(L, T)
            if L >= T:
                # use L originals at distance <=1
                # sum_d[L] = nb*1 + center*0
                cost = nb + 2 * (k - L)
                if cost < ans:
                    ans = cost
            else:
                # must take T originals: sum of T nearest distances
                # gather up to 6 candidate distances
                D = []
                # prev distances
                if prev1[i] != -1:
                    D.append(i - prev1[i])
                if prev2[i] != -1:
                    D.append(i - prev2[i])
                if prev3[i] != -1:
                    D.append(i - prev3[i])
                # next distances
                if nxt1[i] != -1:
                    D.append(nxt1[i] - i)
                if nxt2[i] != -1:
                    D.append(nxt2[i] - i)
                if nxt3[i] != -1:
                    D.append(nxt3[i] - i)
                # sort small list
                D.sort()
                # sum of first T distances
                sumd = D[0] + (D[1] if T > 1 else 0) + (D[2] if T > 2 else 0)
                cost = sumd + 2 * (k - T)
                if cost < ans:
                    ans = cost

        return ans