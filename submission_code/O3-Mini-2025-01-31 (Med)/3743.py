from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        # Compute durations and prefix sum of durations:
        durations = [endTime[i] - startTime[i] for i in range(n)]
        P = [0]*(n+1)
        for i in range(1, n+1):
            P[i] = P[i-1] + durations[i-1]
        total_duration = P[n]
        
        # internal candidate: if we choose two fixed meetings (not necessarily adjacent originally)
        # and plan to move the ones before and after.
        # From some algebra one may show that for fixed indices i and j (i<j) the candidate gap is:
        #      f(i,j) = (startTime[j] - P[j]) - (endTime[i] - P[i+1])
        # and the move budget forces: i + (n - j - 1) <= k.
        #
        # Let δ = n - k - 1 so that j must be at least i + δ.
        δ = n - k - 1  # note: if k is very large, δ may be negative.
        
        internal_candidate = -10**18  # a very small number
        # We will compute arrays A and B:
        # A[j] = startTime[j] - P[j]
        # B[i] = endTime[i] - P[i+1]
        A = [startTime[j] - P[j] for j in range(n)]
        B = [endTime[i] - P[i+1] for i in range(n)]
        
        # We are interested in pairs (i, j) with j - i >= δ.
        # For each valid j, i can range from 0 to j - δ.
        minB = 10**18
        internal_candidate_found = False
        # We iterate j from max(0, δ) to n-1.
        for j in range(max(0, δ), n):
            # update the minimal B for i in [0, j - δ]
            # when j == δ, valid i is just 0.
            if j - δ >= 0:
                # update minB with B[j - δ]
                # (We do it “online”: before processing j, ensure we have the minimum for i in [0, j-δ].)
                if j - δ == 0:
                    minB = B[0]
                else:
                    # ensure that as j increases, we update with the new index (j-δ)
                    minB = min(minB, B[j - δ])
                candidate = A[j] - minB
                if candidate > internal_candidate:
                    internal_candidate = candidate
                    internal_candidate_found = True
        if not internal_candidate_found:
            internal_candidate = -10**18  # if none valid, ignore
        
        # Now consider candidate where the free time is placed at one of the ends.
        # Option 1: free gap at beginning.
        left_candidate = -10**18
        # Suppose we decide that the first fixed meeting is at some index j.
        # Then meetings 0,...,j-1 (if any) are moved and we can pack them at the very beginning.
        # They occupy exactly P[j] time.
        # The fixed meeting j remains at its original start, so the gap before it becomes:
        #    gap = startTime[j] - P[j]
        # The move budget requires that the number of moved meetings in front is j, so j <= k.
        for j in range(min(n, k+1)):
            candidate = startTime[j] - P[j]
            left_candidate = max(left_candidate, candidate)
        
        # Option 2: free gap at the end.
        right_candidate = -10**18
        # Suppose we decide that the last fixed meeting is at index j.
        # Then meetings j+1,..., n-1 are moved and can be packed at the very end.
        # Their total duration is P[n] - P[j+1].
        # The fixed meeting j is at its original end.
        # Hence the gap after it becomes:
        #    gap = (eventTime - (P[n] - P[j+1])) - endTime[j]
        # The move budget requires that the number of moved meetings at end is (n-1 - j) <= k.
        for j in range(n-1, -1, -1):
            if (n - 1 - j) <= k:
                candidate = eventTime - endTime[j] - (P[n] - P[j+1])
                right_candidate = max(right_candidate, candidate)
        
        # Also if k == n, we could move all meetings and pack them arbitrarily.
        all_moved_candidate = -10**18
        if k == n:
            all_moved_candidate = eventTime - total_duration
        
        best = max(internal_candidate, left_candidate, right_candidate, all_moved_candidate)
        # Free time cannot be negative.
        if best < 0:
            best = 0
        return best