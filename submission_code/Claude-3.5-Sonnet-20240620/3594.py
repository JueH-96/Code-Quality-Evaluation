class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # Find the two largest elements
        largest = max(nums)
        second_largest = max(num for num in nums if num != largest)
        
        # Check if the largest element is the outlier
        if total_sum - largest == (n - 2) * (total_sum - largest) / (n - 1):
            return largest
        
        # Check if the second largest element is the outlier
        if total_sum - second_largest == (n - 2) * (total_sum - second_largest) / (n - 1):
            return second_largest
        
        # If neither of the above, the outlier is the smallest element
        return min(nums)