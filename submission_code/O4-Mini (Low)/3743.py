from collections import deque
from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        # durations prefix sum Pdur[i] = sum of durations of meetings [0..i-1]
        Pdur = [0] * (n + 1)
        for i in range(n):
            Pdur[i+1] = Pdur[i] + (endTime[i] - startTime[i])
        total_dur = Pdur[n]
        # If we can move all meetings, we can make them contiguous,
        # so max free is total event minus sum of all durations
        if k >= n:
            return eventTime - total_dur
        
        # Compute static inner gaps g_inner[i] = startTime[i+1] - endTime[i], for i=0..n-2
        # We'll call that array g_inner of length n-1
        g_inner = [0] * (n - 1)
        for i in range(n - 1):
            g_inner[i] = startTime[i+1] - endTime[i]
        
        # Number of fixed meetings
        fixed = n - k
        # We'll slide a window of size (fixed - 1) over g_inner to get the max inner gap
        W = fixed - 1  # window size in g_inner
        # Prepare inner_max list for each window l=0..k
        inner_max = [0] * (k + 1)
        if W > 0:
            dq = deque()
            # standard sliding-window maximum on g_inner with window size W
            for i, val in enumerate(g_inner):
                # pop smaller from back
                while dq and g_inner[dq[-1]] <= val:
                    dq.pop()
                dq.append(i)
                # remove out-of-window from front
                if dq[0] == i - W:
                    dq.popleft()
                # record when we've filled the first window
                if i >= W - 1:
                    l = i - (W - 1)
                    if l <= k:
                        inner_max[l] = g_inner[dq[0]]
        # else if W<=0, inner_max remains all zeros
        
        ans = 0
        # slide over fixed-block start l in [0..k]
        # fixed meetings indices are [l .. r], r = l + fixed - 1
        for l in range(k + 1):
            r = l + fixed - 1
            # left edge free after packing moved meetings next to fixed l
            # moved before l have total duration Pdur[l]
            left_gap = startTime[l] - Pdur[l]
            # right edge free similarly
            # moved after r have duration total_dur - Pdur[r+1]
            moved_after = total_dur - Pdur[r+1]
            right_gap = (eventTime - endTime[r]) - moved_after
            # inner static max gap in this block
            im = inner_max[l]
            # best for this choice of fixed block
            best_here = max(left_gap, right_gap, im)
            ans = max(ans, best_here)
        
        return ans