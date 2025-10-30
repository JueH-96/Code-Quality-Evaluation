class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)

        # Calculate the sum of the first n-2 elements
        sum_special = sum(nums[:n-2])

        # The outlier is the largest element if the sum of the first n-2 elements is not equal to the (n-1)th element
        if sum_special != nums[n-2]:
            return nums[n-1]

        # If the sum of the first n-2 elements is equal to the (n-1)th element, the outlier is the (n-1)th element
        return nums[n-2]