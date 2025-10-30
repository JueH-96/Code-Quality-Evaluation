class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # Initialize two variables to keep track of the longest valid subsequence
        # ending with an even sum and an odd sum respectively
        even_length = 1
        odd_length = 1
        
        # Initialize the result with 1 since the minimum length of any subsequence is 1
        result = 1
        
        # Iterate through the array starting from the second element
        for i in range(1, len(nums)):
            # Calculate the parity of the sum of the current pair
            current_parity = (nums[i] + nums[i - 1]) % 2
            
            if current_parity == 0:
                # If the current pair sum is even, update the even_length
                even_length = odd_length + 1
            else:
                # If the current pair sum is odd, update the odd_length
                odd_length = even_length + 1
            
            # Update the result with the maximum length found so far
            result = max(result, even_length, odd_length)
        
        return result