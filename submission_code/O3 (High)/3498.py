from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2                      # number of symmetric pairs
        
        # cnt_diff[x]  : how many pairs already have |a-b| == x   (0 change needed)
        # cover[x]     : after prefix-sum it will contain how many
        #                pairs can reach difference x with ≤ 1 change
        cnt_diff = [0] * (k + 1)
        cover_diff = [0] * (k + 2)      # +2 so that we may touch index k+1 safely
        
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            
            # 0-change information
            d = abs(a - b)
            cnt_diff[d] += 1
            
            # 1-change information
            # For a fixed element v we can obtain any difference
            # X in [0 , max(v, k-v)] by only modifying the other element.
            hi = max(a, k - a, b, k - b)   # largest X reachable with ≤1 change for this pair
            
            cover_diff[0]      += 1
            cover_diff[hi + 1] -= 1         # (hi + 1) is always ≤ k + 1
            
        # turn difference array into prefix counts => pairs reachable with ≤1 change
        at_most_one = [0] * (k + 1)
        running = 0
        for x in range(k + 1):
            running += cover_diff[x]
            at_most_one[x] = running
        
        # evaluate every possible X, keep the minimum number of changes
        ans = 2 * m                        # worst case: change every element
        for x in range(k + 1):
            # total changes = 2*m  - (#pairs saved by 1 change) - (extra saving for 0 change)
            changes = 2 * m - at_most_one[x] - cnt_diff[x]
            if changes < ans:
                ans = changes
        return ans