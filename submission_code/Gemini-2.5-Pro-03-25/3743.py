import collections
from typing import List

class Solution:
  """
  Finds the maximum possible length of a continuous period of free time 
  after rescheduling at most k meetings. The core idea is that rescheduling 
  meeting `i` effectively allows merging the free time gap immediately before 
  meeting `i` (let's call it gap `i`) and the gap immediately after meeting `i` 
  (gap `i+1`). With `k` reschedules, we can potentially merge a sequence of 
  up to `k+1` adjacent gaps. This means we can consolidate the total free time 
  from these `k+1` gaps into one single large gap. The problem then reduces to 
  finding the maximum sum of a contiguous subarray of the initial gap lengths, 
  where the subarray length is at most `k+1`.
  """
  def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
    """
    Calculates the maximum free time by finding the maximum sum of a contiguous
    subarray of gaps with length at most k+1, using a sliding window approach 
    on prefix sums.

    Args:
      eventTime: The total duration of the event, from time 0 to eventTime.
      k: The maximum number of meetings that can be rescheduled.
      startTime: A list of start times for the n meetings.
      endTime: A list of end times for the n meetings.

    Returns:
      The maximum possible continuous free time achievable after rescheduling
      at most k meetings.
    """
        
    n = len(startTime)
    
    # Problem constraints state n >= 2. 
    # If n were 0, the entire eventTime would be free.
    # If n=1, the logic still applies correctly.
    
    # Calculate initial gaps. There are n+1 potential gaps in total.
    # Let gaps[i] represent the i-th gap.
    # gaps[0]: gap before the first meeting [0, startTime[0]]
    # gaps[i+1]: gap between meeting i and meeting i+1, [endTime[i], startTime[i+1]] for 0 <= i < n-1
    # gaps[n]: gap after the last meeting [endTime[n-1], eventTime]
    gaps = [0] * (n + 1)
    
    # Calculate gap before the first meeting
    gaps[0] = startTime[0] - 0 
    
    # Calculate gaps between consecutive meetings
    for i in range(n - 1):
        # The gap length must be non-negative due to problem constraints: endTime[i] <= startTime[i+1]
        gaps[i+1] = startTime[i+1] - endTime[i]
            
    # Calculate gap after the last meeting
    # The gap length must be non-negative due to problem constraints: endTime[n-1] <= eventTime
    gaps[n] = eventTime - endTime[n-1] 

    num_gaps = n + 1
    
    # Compute prefix sums of the gaps array.
    # prefix_sums[i] stores the sum of gaps G[0]...G[i-1].
    # The array size is num_gaps + 1, with indices ranging from 0 to num_gaps.
    prefix_sums = [0] * (num_gaps + 1)
    for i in range(num_gaps):
        prefix_sums[i+1] = prefix_sums[i] + gaps[i]
            
    max_free_time = 0
    
    # Use a deque (double-ended queue) to efficiently find the minimum prefix sum 
    # within a sliding window. This is used to find the maximum subarray sum 
    # with a length constraint.
    # We want to find max(prefix_sums[j] - prefix_sums[i]) such that the number of gaps 
    # in the sum (which is j-i) is at most k+1.
    # The deque stores indices `i` of the prefix_sums array.
    dq = collections.deque()
    # Initialize the deque with index 0, corresponding to prefix_sums[0] = 0.
    dq.append(0) 
    
    # Iterate `j` from 1 to num_gaps (inclusive). `j` represents the index corresponding 
    # to the end of the potential subarray of gaps (specifically, `j` is `q+1` if the subarray is `G[p...q]`).
    for j in range(1, num_gaps + 1):
        # Remove indices `i` from the left end of the deque if they are outside the window constraint.
        # The length of the subarray G[i...j-1] is `j-i`. We require this length to be at most `k+1`.
        # So, `j - i <= k + 1`, which implies `i >= j - k - 1`.
        # Any index `i` in the deque such that `i < j - k - 1` corresponds to a subarray that is too long.
        while dq and dq[0] < j - k - 1:
            dq.popleft()
        
        # Calculate the sum of the subarray ending at index `j-1` (in the gaps array).
        # This sum is `prefix_sums[j] - prefix_sums[i]`. The minimum `prefix_sums[i]`
        # for the valid window starting indices `i` is given by `prefix_sums[dq[0]]`.
        # We must check if the deque is not empty before accessing its front element.
        if dq:
            # `current_window_sum` is the sum of gaps G[dq[0]...j-1]
            current_window_sum = prefix_sums[j] - prefix_sums[dq[0]]
            max_free_time = max(max_free_time, current_window_sum)
        
        # Maintain the monotonic property of the deque: `prefix_sums` values for indices 
        # in the deque should be non-decreasing from front to back.
        # Remove indices `i'` from the right end if `prefix_sums[i'] >= prefix_sums[j]`.
        # This ensures dq[0] always corresponds to the minimum prefix sum in the current valid window.
        while dq and prefix_sums[dq[-1]] >= prefix_sums[j]:
            dq.pop()
        
        # Add the current index `j` to the deque.
        dq.append(j)
            
    return max_free_time