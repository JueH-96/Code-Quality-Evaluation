class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp = {}

        def solve(i, prev):
            if i == n:
                return 0
            if (i, prev) in dp:
                return dp[(i, prev)]

            ans = 0
            # Option 1: Choose nums1[i]
            if nums1[i] >= prev:
                ans = max(ans, 1 + solve(i + 1, nums1[i]))
            # Option 2: Choose nums2[i]
            if nums2[i] >= prev:
                ans = max(ans, 1 + solve(i + 1, nums2[i]))
            #Option 3: Skip current index.  This is crucial for correctness.
            ans = max(ans, solve(i+1, prev))

            dp[(i, prev)] = ans
            return ans

        return solve(0, -1)