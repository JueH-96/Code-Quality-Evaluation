class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        
        total_sum = sum(nums)
        count = Counter(nums)
        max_outlier = float('-inf')
        
        # Try each element as a potential outlier
        for outlier in nums:
            # Sum of remaining elements after removing outlier
            remaining_sum = total_sum - outlier
            
            # The remaining elements consist of special numbers and their sum
            # If special_sum is the sum of special numbers, then:
            # remaining_sum = special_sum + special_sum = 2 * special_sum
            if remaining_sum % 2 == 0:
                special_sum = remaining_sum // 2
                
                # Check if special_sum exists among remaining elements
                # Create a temporary counter without the outlier
                temp_count = count.copy()
                temp_count[outlier] -= 1
                
                if temp_count[special_sum] > 0:
                    max_outlier = max(max_outlier, outlier)
        
        return max_outlier