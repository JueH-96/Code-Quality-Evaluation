import heapq

class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        # We simulate the original path to get the "orig_gain" at each step,
        # which is the change in Manhattan distance if we do not modify that move.
        # For each move, we can optionally spend one change to ensure the gain is +1.
        # So the "benefit" of spending a change on a move is (1 - orig_gain), whenever orig_gain < 1.
        # We then want, at each prefix, to pick up to k moves among those prefixes with the largest benefits,
        # to add to the sum of original gains, and track the maximum over all prefixes.
        
        # min‐heap to keep the k largest benefits seen so far
        min_heap = []
        sum_benefits = 0
        
        x = 0
        y = 0
        prefix_orig = 0   # sum of orig_gain up to current step
        best = 0
        
        for ch in s:
            # Compute orig_gain for this move
            if ch == 'N':
                # moving +y
                if y >= 0:
                    orig_gain = 1
                else:
                    orig_gain = -1
                y += 1
            elif ch == 'S':
                # moving -y
                if y <= 0:
                    orig_gain = 1
                else:
                    orig_gain = -1
                y -= 1
            elif ch == 'E':
                # moving +x
                if x >= 0:
                    orig_gain = 1
                else:
                    orig_gain = -1
                x += 1
            else:  # 'W'
                # moving -x
                if x <= 0:
                    orig_gain = 1
                else:
                    orig_gain = -1
                x -= 1
            
            # update prefix sum of original gains
            prefix_orig += orig_gain
            
            # compute benefit if we spend a change to force gain = +1
            # only matters if orig_gain < 1
            if orig_gain < 1:
                benefit = 1 - orig_gain  # either 1 or 2
                heapq.heappush(min_heap, benefit)
                sum_benefits += benefit
                # if we've stored more than k benefits, discard the smallest
                if len(min_heap) > k:
                    popped = heapq.heappop(min_heap)
                    sum_benefits -= popped
            
            # current best possible distance at this prefix
            current = prefix_orig + sum_benefits
            if current > best:
                best = current
        
        return best