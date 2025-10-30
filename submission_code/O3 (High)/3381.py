from typing import List

class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        # If k is 0, every non-empty subarray already satisfies the condition
        if k == 0:
            return 1

        n = len(nums)
        ans = n + 1  # larger than any possible subarray length

        for left in range(n):
            cur_or = 0
            for right in range(left, n):
                cur_or |= nums[right]
                if cur_or >= k:
                    ans = min(ans, right - left + 1)
                    # any longer subarray starting at 'left' will be longer,
                    # so we can move to the next 'left'
                    break

        return ans if ans <= n else -1