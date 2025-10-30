from typing import List
class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Precompute prefix sums: prefix[i] = sum(nums[:i])
        prefix = [0]*(n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Precompute nge: next index j > i s.t. nums[j] >= nums[i]
        nge = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            # while stack not empty and nums[i] > nums[stack[-1]]? 
            # Actually, we need the first j such that nums[j] >= nums[i]
            # So pop while stack not empty and nums[stack[-1]] < nums[i]
            while stack and nums[stack[-1]] < nums[i]:
                stack.pop()
            nge[i] = stack[-1] if stack else n
            stack.append(i)
        
        # For a fixed start L and a given ending index R, efficiently compute cost(L,R)
        def cost_for_subarray(L: int, R: int) -> int:
            total_cost = 0
            current_idx = L
            current_baseline = nums[L]
            pos = L + 1
            # We simulate the process:
            while pos <= R:
                # Let p = next greater-or-equal index for current_idx
                p = nge[current_idx]
                # We want to process the segment from pos up to min(R, p-1):
                seg_end = min(R, p - 1)
                # in this segment, all elements are below current_baseline
                if seg_end >= pos:
                    # cost on [pos, seg_end] = (number of elements)*current_baseline - (sum of that segment)
                    seg_cost = (seg_end - pos + 1) * current_baseline - (prefix[seg_end+1] - prefix[pos])
                    total_cost += seg_cost
                # Move pos pointer:
                pos = p
                if pos <= R:
                    # update current_idx and baseline
                    current_idx = pos
                    current_baseline = nums[pos]
                    pos += 1
            return total_cost

        # For fixed L, we need to binary search for maximal R in [L, n-1] such that cost_for_subarray(L, R) <= k.
        ans = 0
        for L in range(n):
            lo, hi = L, n - 1
            valid_R = L - 1  # if no R is valid, then no subarray of length >= 1
            
            while lo <= hi:
                mid = (lo + hi) // 2
                c = cost_for_subarray(L, mid)
                if c <= k:
                    valid_R = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            # The number of valid subarrays starting at L is (valid_R - L + 1)
            ans += (valid_R - L + 1)
        return ans