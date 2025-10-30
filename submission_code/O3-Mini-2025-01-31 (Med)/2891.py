from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # Each index i in nums can become any integer in the interval [nums[i]-k, nums[i]+k].
        # To maximize the beauty, we want to choose a target integer x such that as many intervals
        # as possible contain x. That is, we want the maximum number of intervals overlapping at some point.
        #
        # For each number, we define an interval:
        #   interval[i] = [nums[i] - k, nums[i] + k]
        # Then, the answer is simply the maximum overlap (i.e. maximum number of intervals containing a point x)
        # over all possible x.
        #
        # We can compute this with a sweep-line algorithm by marking a +1 event at the left endpoint of the interval,
        # and a -1 event at the right endpoint + 1 (to account for inclusiveness).
        
        events = []
        for num in nums:
            left = num - k
            right = num + k
            events.append((left, 1))         # Start of interval: add 1.
            events.append((right + 1, -1))     # End of interval (right inclusive): subtract 1 after right.
        
        # Sort the events by coordinate. In case of tie, addition should come before removal
        events.sort(key=lambda x: (x[0], -x[1]))
        
        max_overlap = 0
        current_overlap = 0
        for point, delta in events:
            current_overlap += delta
            max_overlap = max(max_overlap, current_overlap)
        
        return max_overlap