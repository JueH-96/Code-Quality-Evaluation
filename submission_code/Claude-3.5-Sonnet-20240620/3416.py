class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        digit_counts = [0] * 10
        
        # Convert all numbers to strings for easy digit access
        str_nums = [str(num) for num in nums]
        
        # Iterate through each digit position
        for i in range(len(str_nums[0])):
            # Count occurrences of each digit at the current position
            for num in str_nums:
                digit_counts[int(num[i])] += 1
            
            # Calculate differences for the current position
            position_diff = n * (n - 1) - sum(count * (count - 1) for count in digit_counts)
            total_sum += position_diff
            
            # Reset digit counts for the next position
            digit_counts = [0] * 10
        
        return total_sum // 2