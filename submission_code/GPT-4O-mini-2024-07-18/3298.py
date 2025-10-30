class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the occurrences of each number
        count = Counter(nums)
        
        # Create a sorted list of unique numbers
        unique_nums = sorted(count.keys())
        
        max_consecutive = 0
        current_consecutive = 0
        
        # Iterate through the unique numbers
        for i in range(len(unique_nums)):
            # If it's the first number or the difference is 1 or we can increase the previous number
            if i == 0 or unique_nums[i] - unique_nums[i - 1] <= 1:
                current_consecutive += count[unique_nums[i]]
            else:
                # Reset the count for a new sequence
                current_consecutive = count[unique_nums[i]]
            
            # Check if we can increase the previous number
            if i > 0 and unique_nums[i] - unique_nums[i - 1] == 2:
                current_consecutive += count[unique_nums[i - 1]]
            
            # Update the maximum consecutive elements found
            max_consecutive = max(max_consecutive, current_consecutive)
        
        return max_consecutive