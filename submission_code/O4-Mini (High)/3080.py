class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # Compute the AND of the entire array.
        total_and = nums[0]
        for x in nums[1:]:
            total_and &= x
        # If the global AND is > 0, the best you can do is one segment.
        if total_and > 0:
            return 1

        n = len(nums)
        # Precompute suffix ANDs: suffix_and[i] = nums[i] & nums[i+1] & ... & nums[n-1]
        suffix_and = [0] * n
        suffix_and[-1] = nums[-1]
        for i in range(n - 2, -1, -1):
            suffix_and[i] = suffix_and[i + 1] & nums[i]

        ans = 0
        # Use -1 (all bits 1 in two's‐complement) as the initial AND-mask.
        curr_and = -1
        for i, x in enumerate(nums):
            curr_and &= x
            # If the current segment AND drops to zero, and the remaining suffix
            # can still be partitioned (or we're at the end), make a cut.
            if curr_and == 0 and (i == n - 1 or suffix_and[i + 1] == 0):
                ans += 1
                curr_and = -1

        return ans