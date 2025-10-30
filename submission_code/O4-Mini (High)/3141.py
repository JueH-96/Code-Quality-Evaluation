from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        S = sum(nums)
        
        # Helper: sliding window on arr to find min length subarray with sum == need
        def min_window(arr: List[int], need: int) -> int:
            INF = float('inf')
            left = 0
            curr = 0
            best = INF
            for right, v in enumerate(arr):
                curr += v
                # shrink while sum too large
                while curr > need and left <= right:
                    curr -= arr[left]
                    left += 1
                # check match
                if curr == need:
                    length = right - left + 1
                    if length < best:
                        best = length
            return best
        
        # Case 1: total sum of one cycle > target
        # then no full cycle can be used; only need subarrays in two concatenated cycles
        if S > target:
            arr2 = nums + nums
            ans = min_window(arr2, target)
            return ans if ans != float('inf') else -1
        
        # Case 2: S <= target; we may use full cycles
        t = target // S
        r0 = target - t * S
        
        ans = float('inf')
        arr2 = nums + nums
        
        # If target is an exact multiple of S, pure full cycles is a candidate
        if r0 == 0:
            ans = t * n
        else:
            # need extra sum r0 via a short prefix/suffix segment
            L0 = min_window(arr2, r0)
            if L0 != float('inf'):
                ans = min(ans, t * n + L0)
        
        # Also consider using one fewer full cycle, then needing sum = r1 in the ends
        # r1 = target - (t-1)*S = r0 + S
        # (this covers wrap-around segments of sum up to 2*S)
        if t >= 1:
            r1 = r0 + S
            L1 = min_window(arr2, r1)
            if L1 != float('inf'):
                ans = min(ans, (t - 1) * n + L1)
        
        return ans if ans != float('inf') else -1