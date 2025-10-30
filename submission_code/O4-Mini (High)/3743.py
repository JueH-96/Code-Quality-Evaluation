from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        # Build the initial gaps array g of length n+1:
        # g[0] = free before first meeting,
        # g[i] = free between meeting i-1 and i, for 1 <= i <= n-1,
        # g[n] = free after last meeting.
        g = [0] * (n + 1)
        g[0] = startTime[0]              # from time 0 to startTime[0]
        for i in range(1, n):
            g[i] = startTime[i] - endTime[i-1]
        g[n] = eventTime - endTime[n-1]  # from endTime[n-1] to eventTime

        # We can remove up to k meetings in one consecutive block, which
        # frees up k+1 consecutive gaps.  Since all g[i] are non-negative,
        # the best block of up to k+1 gaps is exactly k+1 gaps.
        window_size = k + 1
        # Compute the sum of the first window of size k+1
        curr_sum = sum(g[0 : window_size])
        ans = curr_sum

        # Slide the window of size k+1 across g[0..n]
        # start index i runs from 1 to (n+1) - window_size = n - k
        for i in range(1, n - k + 1):
            # remove g[i-1], add g[i+window_size-1]
            curr_sum += g[i + window_size - 1] - g[i - 1]
            if curr_sum > ans:
                ans = curr_sum

        return ans