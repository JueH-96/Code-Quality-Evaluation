from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        # Build the gap array of length n+1:
        # gap[0] = free before first meeting,
        # gap[i] = startTime[i] - endTime[i-1] for i=1..n-1,
        # gap[n] = free after last meeting.
        gaps = [0] * (n + 1)
        # gap before first
        gaps[0] = startTime[0] - 0
        # middle gaps
        for i in range(1, n):
            gaps[i] = startTime[i] - endTime[i-1]
        # gap after last
        gaps[n] = eventTime - endTime[n-1]
        
        # We can "use" k of the other gaps to add to a chosen gap.
        # That corresponds to taking any consecutive window of size k+1
        # in the gaps[] array and summing it.  The maximum such sum
        # is the answer.
        
        window = k + 1
        # compute prefix sums of gaps
        pref = [0] * (n + 2)
        for i in range(n+1):
            pref[i+1] = pref[i] + gaps[i]
        
        max_sum = 0
        # slide a window of length window over the m=n+1 gaps
        # positions 0..n-window+1 inclusive
        for start in range(0, (n+1) - window + 1):
            s = pref[start+window] - pref[start]
            if s > max_sum:
                max_sum = s
        
        return max_sum