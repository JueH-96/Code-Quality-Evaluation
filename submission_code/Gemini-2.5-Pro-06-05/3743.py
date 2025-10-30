import collections
from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        """
        This problem asks for the maximum continuous free time we can create by rescheduling at most k meetings.
        The key insights are:
        1.  A continuous period of free time corresponds to a "gap". Initially, these gaps are before the first meeting, between any two consecutive meetings, and after the last meeting.
        2.  Rescheduling a meeting `i` that sits between two gaps, `gap_{i}` and `gap_{i+1}`, effectively merges these two gaps. The cost of this merge is one reschedule. The new combined gap has a length equal to the sum of the original two gaps.
        3.  To create the largest possible single gap, we should perform a series of such merges on a contiguous block of original gaps. For example, to enlarge `gap_i`, we can merge it with `gap_{i-1}`, then with `gap_{i-2}`, and so on, and similarly with `gap_{i+1}`, `gap_{i+2}`, etc.
        4.  Since we can perform at most `k` reschedules (merges), we can combine at most `k+1` original contiguous gaps.
        5.  Therefore, the problem reduces to finding the maximum sum of `W` consecutive gaps, for all possible window sizes `W` from 1 to `k+1`.

        Algorithm:
        1.  First, calculate all the initial `n+1` gaps.
            - `gaps[0] = startTime[0]` (gap before the first meeting)
            - `gaps[i] = startTime[i] - endTime[i-1]` for `1 <= i < n` (gaps between meetings)
            - `gaps[n] = eventTime - endTime[n-1]` (gap after the last meeting)
        2.  To efficiently find the maximum sum of any `W` consecutive gaps, we can use prefix sums. Compute the prefix sum array of the `gaps`. Let's call it `prefix_sums`. The sum of gaps from index `l` to `j-1` is `prefix_sums[j] - prefix_sums[l]`.
        3.  The problem is now to find `max(prefix_sums[j] - prefix_sums[l])` over all `l, j` such that `1 <= j-l <= k+1`.
        4.  This can be reformulated as `max_{j} (prefix_sums[j] - min_{l} prefix_sums[l])` where `j-(k+1) <= l < j`.
        5.  This is a classic sliding window minimum problem, which can be solved in O(n) time using a deque. We iterate through `j` from `1` to `n+1`, maintaining a deque of indices `l` corresponding to a monotonically increasing sequence of `prefix_sums` values. This allows us to find the minimum `prefix_sums[l]` for the current window in O(1) time.

        The overall time complexity will be O(n) because calculating gaps, prefix sums, and the sliding window all take linear time. The space complexity is O(n) for storing gaps and prefix sums.
        """
        n = len(startTime)

        # 1. Compute all n+1 initial gaps
        gaps = [0] * (n + 1)
        gaps[0] = startTime[0]
        for i in range(n - 1):
            gaps[i + 1] = startTime[i + 1] - endTime[i]
        gaps[n] = eventTime - endTime[n - 1]

        # 2. Compute prefix sums of the gaps
        # prefix_sums[i] will store sum of first i gaps (gaps[0]...gaps[i-1])
        prefix_sums = [0] * (n + 2)
        for i in range(n + 1):
            prefix_sums[i + 1] = prefix_sums[i] + gaps[i]

        # 3. Use sliding window minimum to find max sum of up to k+1 consecutive gaps
        # We want to find max(prefix_sums[j] - prefix_sums[l]) where 1 <= j-l <= k+1
        # This is equivalent to max_{j=1..n+1} (prefix_sums[j] - min_{l in [j-k-1, j-1]} prefix_sums[l])
        
        max_freetime = 0
        # dq will store indices l from prefix_sums array
        dq = collections.deque()
        dq.append(0) # Start with index 0 in the window

        for j in range(1, n + 2):
            # Remove indices from the left of the deque that are no longer in the window.
            # The window for l is [j-(k+1), j-1].
            if dq and dq[0] < j - (k + 1):
                dq.popleft()
            
            # The minimum prefix_sum in the current window is at the front of the deque.
            # The sum of gaps from index dq[0] to j-1 is prefix_sums[j] - prefix_sums[dq[0]].
            max_freetime = max(max_freetime, prefix_sums[j] - prefix_sums[dq[0]])
            
            # Maintain the monotonically increasing property of the deque.
            # Remove indices from the right whose prefix_sum values are greater than or
            # equal to the current one.
            while dq and prefix_sums[dq[-1]] >= prefix_sums[j]:
                dq.pop()
            
            dq.append(j)
                
        return max_freetime