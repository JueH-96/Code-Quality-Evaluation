from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        # For feasibility: each index must appear at least once in changeIndices,
        # if not then marking that index is impossible.
        # (We can check globally by counting appearances.)
        count = [0] * (n + 1)
        for ci in changeIndices:
            count[ci] += 1
        for i in range(1, n + 1):
            if count[i] == 0:
                return -1

        # Helper function: is it possible to achieve marking for all indices by time s?
        def can_finish_by(s: int) -> bool:
            # For each index from 1 to n, find the LAST occurrence (largest t <= s)
            # where changeIndices[t-1] == i. (Note: changeIndices is 0-indexed while seconds are 1-indexed)
            chosen_times = [0] * (n + 1)  # chosen_times[i] is the chosen marking time for index i.
            for i in range(1, n + 1):
                chosen_times[i] = -1  # not found
            for t in range(1, s+1):
                idx = changeIndices[t-1]
                # update to t (we want the latest possible occurrence)
                chosen_times[idx] = t

            # Check that every index got a marking opportunity
            pairs = []
            for i in range(1, n+1):
                if chosen_times[i] == -1:
                    return False
                pairs.append((chosen_times[i], nums[i-1]))
            
            # sort by marking time (deadline)
            pairs.sort(key=lambda x: x[0])
            
            cumulative = 0
            # For each marking, if it is the kth marking (1-indexed), then
            # available decrement moves in the interval [1, t] is t - k.
            for k, (t, req) in enumerate(pairs, start=1):
                cumulative += req
                if cumulative > (t - k):
                    return False
            # Also, total decrement moves must be no more than s - n.
            if cumulative > (s - n):
                return False
            return True
        
        # Because m is at most 2000, we can simply iterate from 1 to m.
        for s in range(1, m+1):
            if can_finish_by(s):
                return s
        return -1