class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Initialize counters for subsequences where consecutive sums are even and odd
        even_count = 0
        odd_count = 0
        
        # Initialize the current parity of the sum of the subsequence
        current_parity = None
        
        for i in range(len(nums) - 1):
            # Calculate the sum of current and next element
            current_sum = nums[i] + nums[i + 1]
            # Determine the parity of the sum
            sum_parity = current_sum % 2
            
            if current_parity is None:
                # If it's the first pair, initialize the parity
                current_parity = sum_parity
                # Start counting the subsequence length
                current_length = 2
            elif current_parity == sum_parity:
                # If the parity matches the current subsequence parity, increase length
                current_length += 1
            else:
                # If the parity changes, update the max length and reset
                if current_parity == 0:
                    even_count = max(even_count, current_length)
                else:
                    odd_count = max(odd_count, current_length)
                
                # Reset for new subsequence
                current_parity = sum_parity
                current_length = 2
        
        # Final update after the loop
        if current_parity == 0:
            even_count = max(even_count, current_length)
        else:
            odd_count = max(odd_count, current_length)
        
        # Return the maximum length found
        return max(even_count, odd_count)