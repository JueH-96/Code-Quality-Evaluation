from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        # We will choose m fixed meetings (not rescheduled)
        m = n - k
        # if m <= 0, we can reschedule all meetings so that they can be clustered arbitrarily -> free time = eventTime.
        if m <= 0:
            return eventTime

        # The idea: choose m meetings (their indices, in order) to keep fixed.
        # Our candidate answer is:
        #   gap before first fixed meeting: startTime[first_fixed]
        #   gap between consecutive fixed meetings: max_{i} (startTime[i] - endTime[i-1])
        #   gap after last fixed meeting: eventTime - endTime[last_fixed]
        #
        # We wish to choose a subsequence F (of length m) that maximizes the maximum gap.
        # A greedy strategy works: we choose the “leftmost” possible meeting as our first fixed meeting
        # then, for the next fixed meeting we try to skip as many meetings as possible (rescheduling them)
        # if that improves the gap between the fixed meetings.
        #
        # We can precompute for each possible starting index for F[0] what
        # gap at beginning is, and then simulate a greedy selection that maximizes the gap between fixed ones.
        # We try over all possibilities for the first fixed meeting.
        
        best_gap = 0

        # We can use a two-pointer style DP.
        # Let dp[i] be the best candidate for a subsequence that ends at meeting i and has length t.
        # Since we only care about gaps between fixed meetings, we iterate over all possible choices for fixed meetings.
        # Instead we will iterate over possible subsequences in one pass with a “window” that has exactly m fixed meetings.
        # Because these meetings must appear in order, the candidate fixed subsequence is fully determined by choosing
        # an index i for the start and an index j for the end (with exactly m meetings in between).
        #
        # For each valid window of fixed meetings indices [L...R] (with R - L + 1 == m) we can compute:
        #   gap1 = startTime[L] (free time from time 0)
        #   gap2 = eventTime - endTime[R] (free time at end)
        #   inside_gap = max_{i = L...R-1} (startTime[i+1] - endTime[i])
        #
        # And the best candidate for that fixed subsequence is candidate_gap = max(gap1, gap2, inside_gap).
        #
        # We want the maximum candidate_gap over all windows of length m.
        #
        # This is done via a simple sliding window.
        
        # Precompute gap inside for adjacent pairs (only valid for consecutive meetings):
        # gap[i] = startTime[i+1] - endTime[i] for i in 0 to n-2.
        gap_arr = [startTime[i+1] - endTime[i] for i in range(n-1)]
        
        # To quickly get the maximum gap inside a window [L, R] (for L<= i < R),
        # we build a segment tree or use a deque; however, note that n<=1e5 so O(n) per window is too slow.
        # We can precompute sparse table for maximum query in gap_arr.
        # But since we are sliding a window (of fixed length = m) along an array of fixed size,
        # we can compute the maximum sliding window in O(n) time using a deque.
        from collections import deque
        
        # We'll compute max_gap_inside for every window of fixed meetings corresponding to indices L ... L+m-1.
        # We'll slide L from 0 to n - m.
        # For each such window, if window has only one fixed meeting then max inside gap = 0.
        res = 0
        
        # First, handle the case if m == 1:
        if m == 1:
            # then our fixed meeting window is just one meeting.
            # Candidate gap = max( startTime[i], eventTime - endTime[i] )
            for i in range(n):
                candidate = max(startTime[i], eventTime - endTime[i])
                if candidate > res:
                    res = candidate
            return res

        # Compute max sliding window over gap_arr for windows of length (m-1)
        dq = deque()
        # Preprocess first window on gap_arr indices 0 .. (m-2)
        window_size = m - 1
        max_gap_inside = [0] * (n - m + 1)  # for each window starting at L
        for i in range(len(gap_arr)):
            # remove indices out of window from front; window covers i-window_size+1 to i
            while dq and dq[0] < i - window_size + 1:
                dq.popleft()
            # remove smaller
            while dq and gap_arr[dq[-1]] <= gap_arr[i]:
                dq.pop()
            dq.append(i)
            if i >= window_size - 1:
                start_index = i - window_size + 1
                # only record if window fits into gap_arr corresponding to fixed window L...(L+m-1)
                if start_index <= n - m:
                    max_gap_inside[start_index] = gap_arr[dq[0]]
        
        # Now slide fixed window over indices L ... L+m-1
        for L in range(n - m + 1):
            R = L + m - 1
            # free gap before first fixed
            candidate1 = startTime[L]
            # free gap after last fixed
            candidate2 = eventTime - endTime[R]
            # free gap inside (if m>=2)
            candidate3 = max_gap_inside[L] if m - 1 > 0 else 0
            candidate = max(candidate1, candidate2, candidate3)
            if candidate > res:
                res = candidate
        return res