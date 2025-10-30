class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        # Sort the numbers in descending order of their least significant bit (LSB)
        nums.sort(key=lambda x: x & -x)
        
        # Initialize the result with the bitwise OR of all numbers
        result = 0
        for num in nums:
            result |= num
        
        # Try to maximize the result by doubling the smallest LSB numbers k times
        for num in nums:
            if k <= 0:
                break
            # Calculate the number of times we can double the current number
            times = min(k, (num & -num).bit_length() - 1)
            # Update the result with the new value
            result |= num << times
            # Decrease the remaining operations
            k -= times
        
        return result