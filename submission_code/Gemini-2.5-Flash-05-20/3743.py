from typing import List

class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)

        # Step 1: Calculate the lengths of all initial free gaps.
        # There are n+1 potential free gaps.
        # g_0: from 0 to startTime[0]
        # g_i: from endTime[i-1] to startTime[i] for i=1..n-1
        # g_n: from endTime[n-1] to eventTime
        
        gaps = []
        
        # Gap before the first meeting
        # This gap length can be 0 if startTime[0] is 0.
        gaps.append(startTime[0] - 0)

        # Gaps between meetings
        # These gaps lengths can be 0 if endTime[i] == startTime[i+1].
        for i in range(n - 1):
            gaps.append(startTime[i+1] - endTime[i])
        
        # Gap after the last meeting
        # This gap length can be 0 if endTime[n-1] is eventTime.
        gaps.append(eventTime - endTime[n-1])

        # Step 2: Initialize max_free_time with the largest single gap.
        # This covers cases where no meetings are moved, or where moving a meeting
        # results in smaller combined gaps than an existing single gap.
        max_free_time = 0
        for gap_len in gaps:
            max_free_time = max(max_free_time, gap_len)
        
        # Step 3: Use a sliding window to find the maximum sum of consecutive gaps.
        # The window [left, right] represents a sequence of gaps (gaps[left] through gaps[right]).
        # To merge these (right - left + 1) gaps into one continuous free period,
        # we need to "absorb" the meetings that separate them.
        # The number of meetings to absorb is (right - left).
        # We are allowed to absorb at most k meetings. So, (right - left) <= k.

        current_gap_sum = 0
        left = 0 # Left pointer for the window of gaps

        # Iterate right pointer for the window of gaps
        # `right` goes from 0 to n (inclusive), representing indices of `gaps`.
        for right in range(len(gaps)): 
            current_gap_sum += gaps[right]
            
            # `num_meetings_to_absorb` is the count of meetings (separators) between `gaps[left]` and `gaps[right]`.
            # For a window `gaps[left...right]`, these are meetings `M_left, M_{left+1}, ..., M_{right-1}`.
            # The count is `right - left`.
            num_meetings_to_absorb = right - left 

            # If the number of meetings we need to absorb exceeds our budget `k`,
            # we must shrink the window from the left.
            while num_meetings_to_absorb > k:
                current_gap_sum -= gaps[left]
                left += 1
                # Recalculate meetings to absorb with the new `left` pointer.
                num_meetings_to_absorb = right - left
            
            # At this point, the window `[left, right]` is valid: `num_meetings_to_absorb <= k`.
            # The `current_gap_sum` represents the total length of free time if these gaps are combined.
            max_free_time = max(max_free_time, current_gap_sum)
            
        return max_free_time