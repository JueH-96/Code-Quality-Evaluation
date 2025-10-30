import collections
from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        
        # 1. Calculate initial gap lengths g_0, ..., g_n
        # g_0: Free time before the first meeting [0, startTime[0]]
        # g_i (1 <= i <= n-1): Free time between meeting i-1 and meeting i [endTime[i-1], startTime[i]]
        # g_n: Free time after the last meeting [endTime[n-1], eventTime]
        
        gaps = []
        
        # Gap before the first meeting
        gaps.append(startTime[0])
        
        # Gaps between meetings
        for i in range(1, n):
            gaps.append(startTime[i] - endTime[i-1])
            
        # Gap after the last meeting
        gaps.append(eventTime - endTime[n-1])
        
        N = n + 1 # Total number of initial gaps
        
        # 2. Calculate prefix sums of gaps
        # prefix_sum[i] = sum(gaps[0] ... gaps[i-1])
        # prefix_sum[0] = 0
        # prefix_sum[i+1] = prefix_sum[i] + gaps[i] for i = 0 to N-1
        prefix_sum = [0] * (N + 1)
        for i in range(N):
            prefix_sum[i+1] = prefix_sum[i] + gaps[i]
            
        # 3. Use sliding window minimum (deque) on prefix sums
        # We want to find the maximum sum of a contiguous subarray of gaps, gaps[i...j].
        # This sum is prefix_sum[j+1] - prefix_sum[i].
        # The gaps involved are gaps[i], gaps[i+1], ..., gaps[j]. The number of gaps is j - i + 1.
        # The meetings that need to be potentially shifted to merge these gaps are M_i, M_{i+1}, ..., M_{j-1}.
        # The number of meetings is j - i.
        # We can merge gaps[i...j] using j - i shifts.
        # Constraint: j - i <= k. This implies i >= j - k.
        # Also, the indices i and j must be valid indices in the gaps array: 0 <= i <= j < N.
        
        # We iterate through all possible end indices j (in gaps array) from 0 to N-1.
        # The corresponding index in the prefix_sum array is j+1. Let curr = j+1.
        # curr runs from 1 to N.
        # For a fixed curr (representing the end of the gap block in prefix_sum),
        # we need to find the minimum prefix_sum[i] where i is the start index in prefix_sum.
        # The index i in prefix_sum corresponds to the start index i in the gaps array.
        # The end index in gaps is curr - 1.
        # The start index in gaps is i.
        # The number of meetings between gaps[i] and gaps[curr-1] is (curr-1) - i.
        # This number must be <= k. So, (curr-1) - i <= k  => i >= curr - 1 - k.
        # Also, the index i in prefix_sum must be non-negative: i >= 0.
        # And the index i in prefix_sum must be less than curr: i < curr.
        # So the valid indices i for prefix_sum[i] are in the range [max(0, curr - 1 - k), curr - 1].
        
        max_free_time = 0
        
        dq = collections.deque()
        
        # Iterate through all possible end indices in the prefix_sum array (from P[0] to P[N])
        # curr represents the current index in the prefix_sum array.
        # prefix_sum[curr] is the sum up to gaps[curr-1].
        for curr in range(N + 1):
            # Remove indices from the front of the deque that are outside the valid window for the start index i.
            # The valid window for index i in prefix_sum is [max(0, curr - 1 - k), curr - 1].
            window_start_idx_for_P = max(0, curr - 1 - k)
            
            # If the index at the front of the deque is less than the window start, it's too old.
            while dq and dq[0] < window_start_idx_for_P:
                 dq.popleft()
                 
            # prefix_sum[dq[0]] is the minimum prefix_sum[i] among the valid start indices i in the window.
            # If the deque is not empty, calculate the potential max free time ending at gaps[curr-1].
            # This is prefix_sum[curr] - prefix_sum[dq[0]].
            # This represents the sum of gaps from index dq[0] to curr-1 in the gaps array.
            if dq: # dq will contain at least one element if curr >= window_start_idx_for_P
                 max_free_time = max(max_free_time, prefix_sum[curr] - prefix_sum[dq[0]])
            
            # Maintain the monotonic property of the deque: store indices i such that prefix_sum[i] is increasing.
            # If prefix_sum[dq.back()] >= prefix_sum[curr], the index dq.back() is not a useful minimum anymore
            # for any future window starting after dq.back() (since prefix sums are non-decreasing, prefix_sum[curr]
            # is a better or equal minimum).
            while dq and prefix_sum[dq[-1]] >= prefix_sum[curr]:
                 dq.pop()
                 
            # Add the current index `curr` to the deque. It's a potential start index `i` for future windows.
            dq.append(curr)
            
        return max_free_time