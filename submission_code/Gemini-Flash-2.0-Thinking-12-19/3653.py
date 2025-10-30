from typing import List

class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = -float('inf')

        for i in range(n):
            for j in range(i, n):
                sub_array = nums[i:j+1]
                sub_array_len = len(sub_array)
                if sub_array_len % k == 0:
                    current_sum = sum(sub_array)
                    max_sum = max(max_sum, current_sum)

        if max_sum == -float('inf'):
            # Handle the case where no subarray of size divisible by k exists.
            # In this problem, it seems there is always at least one subarray with size divisible by k,
            # for example, if k=1, any single element subarray has size divisible by k.
            # However, to be safe, we can return the maximum element if no valid subarray is found.
            # Or, based on examples, if no valid subarray sum is positive, return the largest single element sum
            # which would be the largest element itself if it exists or negative inf if array is empty.
            # But based on constraints, nums is not empty, and k >= 1, so there will always be a valid subarray.
            # So, if max_sum is still -inf, it means all valid subarray sums are negative.
            # In that case, we need to find the maximum among all possible valid subarray sums,
            # even if they are negative. The initialization with -inf should handle this case correctly.
            # Let's check the examples again. Example 2 has all negative numbers and k=4, output is -10.
            # So -inf initialization is correct. We just need to handle the case where no valid subarray exists,
            # although based on problem description, it's always possible, for example if k=1, any subarray will work.
            # Let's assume there is always at least one valid subarray, and -inf initialization works fine.

            # If after all iterations, max_sum is still -inf, it means there was no valid subarray.
            # However, based on the problem constraints, it is always possible to have subarrays with length divisible by k.
            # For example, if k = 1, any subarray has length divisible by k.
            # If k > 1, and array length is n, if n >= k, we can have subarrays of length k.
            # If n < k, and k > 1, it's impossible to have subarray of length divisible by k unless k = 1.
            # But constraint is 1 <= k <= nums.length. So nums.length >= k >= 1.
            # So there will always be subarrays of length divisible by k.
            # If all subarray sums are negative, max_sum will be the largest negative sum, which is correct.

            if max_sum == -float('inf'):
                # This condition should ideally not be hit given the problem constraints.
                # But just in case, return the largest single element if all valid subarray sums are negative.
                max_single_element = -float('inf')
                for x in nums:
                    max_single_element = max(max_single_element, x)
                return max_single_element if max_single_element != -float('inf') else 0 # return 0 if nums is empty? but nums is not empty

                # Actually, if there is always a valid subarray, and max_sum is -inf at the end,
                # it means all valid subarray sums are negative infinity, which is not possible.
                # So, the -inf initialization should work correctly even for all negative numbers.
                # Let's just return max_sum as is, it should be the correct answer even for all negative numbers cases.
                pass # remove this pass, and just return max_sum

        return max_sum