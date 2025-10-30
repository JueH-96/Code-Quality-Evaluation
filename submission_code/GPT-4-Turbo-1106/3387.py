class Solution:
    def minOperationsToMakeMedianK(self, nums: list[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        median_index = (n - 1) // 2
        operations = 0
        
        # If the array has an odd number of elements, there's only one median
        if n % 2 == 1:
            return abs(nums[median_index] - k)
        
        # If the array has an even number of elements, the median is the larger of the two middle elements
        else:
            # If k is between the two middle elements, no operations are needed
            if nums[median_index] <= k <= nums[median_index + 1]:
                return 0
            # If k is less than the smaller middle element
            elif k < nums[median_index]:
                for i in range(median_index, -1, -1):
                    if nums[i] > k:
                        operations += nums[i] - k
                    else:
                        break
            # If k is greater than the larger middle element
            else:
                for i in range(median_index + 1, n):
                    if nums[i] < k:
                        operations += k - nums[i]
                    else:
                        break
            return operations

# Example usage:
# sol = Solution()
# print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 4))  # Output: 2
# print(sol.minOperationsToMakeMedianK([2,5,6,8,5], 7))  # Output: 3
# print(sol.minOperationsToMakeMedianK([1,2,3,4,5,6], 4))  # Output: 0