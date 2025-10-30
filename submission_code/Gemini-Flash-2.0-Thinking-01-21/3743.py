from typing import List
from collections import deque

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        
        # 1. Calculate initial gap lengths
        # The free time intervals are [0, startTime[0]), [endTime[i-1], startTime[i]) for i=1..n-1, and [endTime[n-1], eventTime].
        # Let's represent these gap lengths.
        
        gaps = []
        
        # Gap before the first meeting [0, startTime[0))
        gaps.append(startTime[0])
        
        # Gaps between meetings [endTime[i-1], startTime[i)) for i=1..n-1
        for i in range(n - 1):
            gaps.append(startTime[i+1] - endTime[i])
            
        # Gap after the last meeting [endTime[n-1], eventTime]
        gaps.append(eventTime - endTime[n-1])
        
        # Number of gaps is n + 1
        num_gaps = len(gaps) 
        
        # 2. Calculate prefix sums of gap lengths
        # P[x] = sum(gaps[0...x-1]) for x = 0 to num_gaps
        # P[0] = 0
        # P[1] = gaps[0]
        # P[i] = sum(gaps[0...i-1])
        P = [0] * (num_gaps + 1)
        for i in range(num_gaps):
            P[i+1] = P[i] + gaps[i]
            
        # 3. Find the maximum sum of a contiguous subarray of gaps with length at most k+1.
        # A contiguous subarray of gaps [gaps[i], gaps[i+1], ..., gaps[j]] has length j - i + 1.
        # Merging these gaps requires removing the meetings M_i, M_{i+1}, ..., M_{j-1}.
        # The number of meetings is (j-1) - i + 1 = j - i.
        # We can merge gaps[i...j] if the number of meetings j-i is at most k.
        # So we need the maximum sum sum(gaps[i...j]) for 0 <= i <= j < num_gaps such that j - i <= k.
        
        # sum(gaps[i...j]) = P[j + 1] - P[i]. Indices i, j are 0-based in the gaps array.
        # We want max_{0 <= i <= j < num_gaps, j - i <= k} (P[j+1] - P[i]).
        # Let end_p_idx = j + 1. Then 1 <= end_p_idx <= num_gaps. j = end_p_idx - 1.
        # Let start_p_idx = i. Then 0 <= start_p_idx < num_gaps.
        # The constraint j - i <= k becomes (end_p_idx - 1) - start_p_idx <= k, which means start_p_idx >= end_p_idx - k - 1.
        # Also, start_p_idx <= j = end_p_idx - 1.
        # So, for a fixed end_p_idx, the range for start_p_idx is [max(0, end_p_idx - k - 1), end_p_idx - 1].
        # We need max_{1 <= end_p_idx <= num_gaps} (P[end_p_idx] - min_{max(0, end_p_idx - k - 1) <= start_p_idx < end_p_idx} P[start_p_idx]).
        # We can find the minimum P[start_p_idx] in the sliding window [max(0, end_p_idx - k - 1), end_p_idx - 1] efficiently using a deque.
        
        max_free = 0
        dq = deque()
        
        # Add index 0 (for P[0]) to the deque initially. P[0] = 0, which is the sum before the first gap.
        dq.append(0)
        
        # Iterate end_p_idx from 1 to num_gaps (inclusive).
        # P[end_p_idx] = sum(gaps[0...end_p_idx-1]).
        for end_p_idx in range(1, num_gaps + 1):
            # The window of valid start indices `start_p_idx` for P[start_p_idx] is [max(0, end_p_idx - k - 1), end_p_idx - 1].
            # Indices `start_p_idx` in the deque must be >= max(0, end_p_idx - k - 1).
            window_start_p_start = max(0, end_p_idx - k - 1)
            while dq and dq[0] < window_start_p_start:
                dq.popleft()
            
            # The index dq[0] now holds the index `start_p_idx` that minimizes P[start_p_idx] in the current window.
            # P[end_p_idx] - P[dq[0]] is the sum of gaps from index dq[0] to end_p_idx - 1 (in the gaps array).
            max_free = max(max_free, P[end_p_idx] - P[dq[0]])
            
            # Add current index p_end to the deque, maintaining the increasing property of P values.
            # Indices in dq `q` store `q` such that P[q]` is increasing.
            # We are adding index `p_end` corresponding to `P[p_end]`.
            while dq and P[dq[-1]] >= P[end_p_idx]:
                dq.pop()
            dq.append(end_p_idx)
            
        return max_free