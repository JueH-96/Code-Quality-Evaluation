class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # Initialize the maximum length of the subset
        max_length = 0
        
        # Iterate over each unique number in nums
        for num in freq:
            # Calculate the maximum possible length for the current number
            # We can use the number itself and its square if it exists in the array
            count = freq[num]
            square = num * num
            if square in freq:
                # If the square exists, we can form a pattern [num, square, num]
                count += 1
            
            # Update the maximum length
            max_length = max(max_length, count)
        
        return max_length