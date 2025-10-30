class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        # Initialize scores based on the parity of the first element
        if nums[0] % 2 == 0:  # Even
            even_score = nums[0]
            odd_score = -float('inf')
        else:  # Odd
            odd_score = nums[0]
            even_score = -float('inf')
        
        # Process each position from index 1 onward
        for i in range(1, len(nums)):
            if nums[i] % 2 == 0:  # Current number is even
                # Update max score ending with even number
                even_score = nums[i] + max(even_score, odd_score - x)
            else:  # Current number is odd
                # Update max score ending with odd number
                odd_score = nums[i] + max(odd_score, even_score - x)
        
        # Return the maximum of both parity-ending scores
        return max(even_score, odd_score)