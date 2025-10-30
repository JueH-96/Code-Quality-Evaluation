from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)

        # Build the list of gaps  g[0 .. n]
        gaps = [0]*(n+1)
        gaps[0] = startTime[0]                      # free time before first meeting
        for i in range(1, n):
            gaps[i] = startTime[i] - endTime[i-1]   # free time between consecutive meetings
        gaps[n] = eventTime - endTime[-1]           # free time after last meeting

        max_len = 0
        window_sum = 0
        left = 0
        max_width = k + 1                           # at most k gaps can be removed → window of ≤ k+1 gaps

        for right in range(len(gaps)):
            window_sum += gaps[right]

            # keep the window length within the allowed width
            while right - left + 1 > max_width:
                window_sum -= gaps[left]
                left += 1

            max_len = max(max_len, window_sum)

        return max_len