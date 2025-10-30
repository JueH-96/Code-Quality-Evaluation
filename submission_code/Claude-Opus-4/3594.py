class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Calculate total sum
        total_sum = sum(nums)
        
        # Count frequency of each number
        freq = Counter(nums)
        
        max_outlier = float('-inf')
        
        # Try each element as potential outlier
        for i, outlier in enumerate(nums):
            # Calculate what the sum of special numbers should be
            remaining_sum = total_sum - outlier
            
            # If remaining_sum is odd, it can't be 2*T
            if remaining_sum % 2 != 0:
                continue
                
            special_sum = remaining_sum // 2
            
            # Check if special_sum exists in the array (excluding current outlier)
            # We need to temporarily reduce the count of outlier
            freq[outlier] -= 1
            
            if freq[special_sum] > 0:
                max_outlier = max(max_outlier, outlier)
            
            # Restore the count
            freq[outlier] += 1
        
        return max_outlier