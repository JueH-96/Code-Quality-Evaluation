class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        max_length = 0
        
        # Iterate through each unique number in nums
        for num in freq:
            # Initialize the current length of the sequence
            current_length = 0
            current_num = num
            
            # Check if the sequence can be formed
            while current_num in freq and freq[current_num] > 0:
                current_length += 1
                freq[current_num] -= 1
                current_num *= 2
            
            # Update the maximum length found
            max_length = max(max_length, current_length)
            
            # Restore the frequencies for the next iteration
            current_num = num
            while current_length > 0:
                freq[current_num] += 1
                current_num *= 2
                current_length -= 1
        
        return max_length