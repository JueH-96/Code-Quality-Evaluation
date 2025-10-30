class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        n = len(nums)
        count = 0
        
        # Create a dictionary to store the count of each remainder
        remainder_count = [0] * modulo
        remainder_count[0] = 1  # Initialize the count of remainder 0 to 1
        
        # Iterate through the array
        curr_remainder = 0
        for num in nums:
            # Update the current remainder
            curr_remainder = (curr_remainder + num % modulo) % modulo
            
            # If the current remainder is equal to k, increment the count
            if curr_remainder == k:
                count += 1
            
            # Add the count of the complement remainder to the count
            count += remainder_count[(curr_remainder - k) % modulo]
            
            # Update the count of the current remainder
            remainder_count[curr_remainder] += 1
        
        return count