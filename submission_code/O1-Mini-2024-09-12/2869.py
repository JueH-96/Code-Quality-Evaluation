class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0

        dp1_prev = 1  # Length when choosing nums1[i-1]
        dp2_prev = 1  # Length when choosing nums2[i-1]
        max_length = 1

        for i in range(1, n):
            current1 = nums1[i]
            current2 = nums2[i]
            new_dp1 = new_dp2 = 1  # Reset for current position

            # Update dp1 for choosing nums1[i]
            if current1 >= nums1[i - 1]:
                new_dp1 = max(new_dp1, dp1_prev + 1)
            if current1 >= nums2[i - 1]:
                new_dp1 = max(new_dp1, dp2_prev + 1)

            # Update dp2 for choosing nums2[i]
            if current2 >= nums1[i - 1]:
                new_dp2 = max(new_dp2, dp1_prev + 1)
            if current2 >= nums2[i - 1]:
                new_dp2 = max(new_dp2, dp2_prev + 1)

            # Update the maximum length found so far
            max_length = max(max_length, new_dp1, new_dp2)

            # Set the previous dp values for the next iteration
            dp1_prev, dp2_prev = new_dp1, new_dp2

        return max_length