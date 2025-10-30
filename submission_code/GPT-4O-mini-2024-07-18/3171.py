class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(x for x in nums1 if x > 0)
        sum2 = sum(x for x in nums2 if x > 0)
        count1 = nums1.count(0)
        count2 = nums2.count(0)

        # Calculate the difference in sums
        diff = abs(sum1 - sum2)

        # If the sums are already equal, return the total sum
        if diff == 0:
            return sum1 + sum2

        # If the difference is greater than the total number of zeros we can fill
        # with at least 1, it's impossible to balance the sums
        if diff > count1 + count2:
            return -1

        # Calculate the minimum equal sum we can obtain
        # We need to fill the zeros with at least 1
        # The minimum sum we can achieve is the maximum of the two sums plus the difference
        min_equal_sum = max(sum1, sum2) + diff

        return min_equal_sum