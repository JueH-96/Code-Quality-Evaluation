class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # Calculate the total sum of the array
        total_sum = sum(nums)
        
        # Iterate through the list to find the largest potential outlier
        largest_outlier = float('-inf')
        
        for num in nums:
            # Calculate the sum of the special numbers if `num` is considered as the sum of special numbers
            sum_specials = total_sum - num
            
            # Check if sum_specials can be formed by other elements in the array
            if sum_specials % 2 == 0:
                # The sum of specials should be even to be split into two integers
                half_sum_specials = sum_specials // 2
                
                # Check if half_sum_specials can be formed
                if half_sum_specials in freq:
                    # If half_sum_specials is the same as num, we need at least two occurrences
                    if half_sum_specials == num and freq[num] < 2:
                        continue
                    # If half_sum_specials is different from num, we need at least one occurrence
                    else:
                        # We found a valid configuration, check for potential outlier
                        for x in nums:
                            if x != num and x != half_sum_specials:
                                largest_outlier = max(largest_outlier, x)
        
        return largest_outlier