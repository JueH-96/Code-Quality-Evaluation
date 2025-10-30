from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        # Sort the array to work with contiguous windows
        nums.sort()
        n = len(nums)
        # Prefix sums for quick range-sum queries
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]

        res = 1
        l = 0
        # Slide the right end of the window
        for r in range(n):
            # Shrink from the left until the cost to equalize
            # all nums[l..r] to their median is <= k
            while l <= r:
                m = r - l + 1
                mid = l + m // 2
                med = nums[mid]
                # Cost to bring left half [l..mid-1] up to med
                cost_left = med * (mid - l) - (prefix[mid] - prefix[l])
                # Cost to bring right half [mid+1..r] down to med
                cost_right = (prefix[r + 1] - prefix[mid + 1]) - med * (r - mid)
                if cost_left + cost_right <= k:
                    break
                l += 1
            # Update the maximum window size
            res = max(res, r - l + 1)

        return res