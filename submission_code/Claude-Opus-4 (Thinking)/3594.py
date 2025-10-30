class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        
        total_sum = sum(nums)
        count = Counter(nums)
        
        max_outlier = float('-inf')
        
        for num in nums:
            # Assume num is the outlier
            if (total_sum - num) % 2 == 0:
                expected_sum = (total_sum - num) // 2
                
                # Check if expected_sum exists in the array
                if num == expected_sum:
                    # Need at least 2 occurrences (one for outlier, one for sum)
                    if count[num] >= 2:
                        max_outlier = max(max_outlier, num)
                else:
                    if count[expected_sum] > 0:
                        max_outlier = max(max_outlier, num)
        
        return max_outlier