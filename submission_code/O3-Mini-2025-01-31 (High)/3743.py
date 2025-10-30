from collections import deque
from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        # durations and prefix sums
        durations = [endTime[i] - startTime[i] for i in range(n)]
        P = [0] * (n + 1)
        for i in range(n):
            P[i+1] = P[i] + durations[i]
        total_duration = P[n]
        
        # Special case: if we can move all meetings then they can be packed continuously,
        # and the free time equals eventTime - total_duration.
        if k == n:
            return eventTime - total_duration
        
        # It turns out that one may “prove” that it suffices to maximize
        # one of the following candidates:
        # 1. Front candidate: consider fixing a meeting as the first one.
        #    For an index i (which means we “move” the meetings 0..i–1),
        #    the gap is: startTime[i] - (sum of durations of meetings 0..i–1)
        #    i.e. X[i] = startTime[i] – P[i].
        X = [startTime[i] - P[i] for i in range(n)]
        # 2. Back candidate: if we fix a meeting as the last one.
        #    For an index j (which forces moving meetings j+1..n–1),
        #    a candidate gap is (eventTime – (total_duration – P[j+1])) – endTime[j].
        # 3. Internal candidate: Suppose we want to “bridge over” some meetings.
        #    If we fix two meetings from positions i and j (with i < j) and move the ones in between,
        #    the gap will be: startTime[j] – endTime[i] – (sum of durations for meetings i+1..j–1).
        #    One may rewrite this as: (startTime[j] – P[j]) – (endTime[i] – P[i+1])
        #    i.e. using X[j] – Y[i] where we define Y[i] = endTime[i] – P[i+1].
        Y = [endTime[i] - P[i+1] for i in range(n)]
        
        # candidate_front: over indices i with i <= k (since moving i meetings uses i moves)
        candidate_front = -10**18
        for i in range(min(n, k+1)):
            candidate_front = max(candidate_front, X[i])
        
        # candidate_end: for indices j such that (n-1 - j) <= k (i.e. j is “late” enough)
        candidate_end = -10**18
        start_j = max(0, n - 1 - k)
        for j in range(start_j, n):
            # When fixing meeting j as the last meeting, we “move” those after j;
            # the gap available at the end is:
            #  (eventTime - (total_duration - P[j+1])) - endTime[j]
            candidate_end = max(candidate_end, (eventTime - (total_duration - P[j+1])) - endTime[j])
        
        # candidate_internal: for j>=1, we want to choose an i from a “window”
        # (we are allowed to “bridge” at most k meetings, i.e. we require that
        # j - i - 1 <= k)
        # Notice that for fixed j, and for any i with max(0, j-k-1) <= i < j,
        # the candidate is X[j] - Y[i]. (When i = j-1, no meeting is “bridged”.
        # When i < j-1, we are “skipping over” some meetings.)
        candidate_internal = -10**18
        dq = deque()  # will hold indices i (for Y) in the window; we maintain Y[i] increasing.
        # We iterate j from 0 to n-1 and update our sliding–window:
        for j in range(n):
            # The window of i that are allowed when considering candidate for j
            # is all i with i >= max(0, j - k - 1) and i < j.
            while dq and dq[0] < j - k - 1:
                dq.popleft()
            if j > 0 and dq:
                candidate_internal = max(candidate_internal, X[j] - Y[dq[0]])
            # update the deque with index j (so that for future j's,
            # i = j is in the window) while keeping window increasing in Y's.
            while dq and Y[dq[-1]] >= Y[j]:
                dq.pop()
            dq.append(j)
        
        # Also note that if no meeting is moved (k=0) the free time is just
        # the original gap (which appears among the gap before the first meeting,
        # after the last meeting, or between two meetings).
        candidate_original = startTime[0]  # gap at the very beginning
        candidate_original = max(candidate_original, eventTime - endTime[n-1])
        for i in range(n - 1):
            candidate_original = max(candidate_original, startTime[i+1] - endTime[i])
        
        candidate_all = max(candidate_front, candidate_end, candidate_internal, candidate_original)
        return candidate_all