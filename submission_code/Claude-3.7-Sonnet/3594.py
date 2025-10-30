class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        from collections import Counter
        freq = Counter(nums)
        
        max_outlier = float('-inf')
        
        for val in set(nums):  # Try each value as potential sum
            potential_outlier = total_sum - 2 * val
            
            # Check if this potential outlier is valid
            if val == potential_outlier:
                # If the potential sum and outlier are the same value,
                # we need at least 2 occurrences of that value
                if freq[val] >= 2:
                    max_outlier = max(max_outlier, potential_outlier)
            else:
                # If they're different values, just check if the outlier exists
                if freq[potential_outlier] >= 1:
                    max_outlier = max(max_outlier, potential_outlier)
        
        return max_outlier