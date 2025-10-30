from collections import deque
from typing import List

class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        """
        We use a sliding window [l..r]. For each position i in the window,
        its final value f[i] = max(nums[l..i]). The cost to turn nums[l..r]
        into non-decreasing is sum(f[i] - nums[i]) over i in [l..r].
        
        We maintain:
          - A deque of segments (M, cnt), where M is a distinct running max
            that appears cnt times as f[i] for consecutive positions.
          - The current total cost.
        As we move r right, we compute the added cost and update the last segment.
        As we advance l (to shrink when cost>k), we subtract the contribution of nums[l]
        from the first segment, and pop it if its count drops to zero.
        """
        n = len(nums)
        segs = deque()   # each element is [M, cnt]
        cost = 0
        res = 0
        l = 0
        
        for r in range(n):
            x = nums[r]
            # Add position r
            if not segs:
                # first element in window
                segs.append([x, 1])
            else:
                M_last, cnt_last = segs[-1]
                if x >= M_last:
                    # new running max
                    segs.append([x, 1])
                else:
                    # raises up to M_last
                    segs[-1][1] += 1
                    cost += (M_last - x)
            
            # If cost too big, advance l
            while cost > k:
                M_first, cnt_first = segs[0]
                # removing position l incurs reversing its increment
                cost -= (M_first - nums[l])
                # decrement count
                segs[0][1] -= 1
                if segs[0][1] == 0:
                    segs.popleft()
                l += 1
            
            # All subarrays ending at r starting in [l..r] are valid
            res += (r - l + 1)
        
        return res