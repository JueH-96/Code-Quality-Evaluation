class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1_no_zeros = sum(n for n in nums1 if n != 0)
        n1_zeros = nums1.count(0)
        sum2_no_zeros = sum(n for n in nums2 if n != 0)
        n2_zeros = nums2.count(0)

        if n1_zeros == 0 and n2_zeros == 0:
            return sum1_no_zeros if sum1_no_zeros == sum2_no_zeros else -1

        min_possible_sum1 = sum1_no_zeros + n1_zeros
        min_possible_sum2 = sum2_no_zeros + n2_zeros

        min_equal_sum = max(min_possible_sum1, min_possible_sum2)

        # Check if the min_equal_sum is achievable
        if min_equal_sum > sum1_no_zeros and n1_zeros == 0:
            return -1
        if min_equal_sum > sum2_no_zeros and n2_zeros == 0:
            return -1

        return min_equal_sum