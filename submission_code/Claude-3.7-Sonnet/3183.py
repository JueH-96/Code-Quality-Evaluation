class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        bit_count = [0] * 31  # Counts for bit positions 0 to 30
        
        # Count how many numbers have each bit set
        for num in nums:
            for i in range(31):
                if num & (1 << i):  # Check if bit i is set in num
                    bit_count[i] += 1
        
        # Calculate the K-or
        result = 0
        for i in range(31):
            if bit_count[i] >= k:
                result |= (1 << i)  # Set bit i in the result
        
        return result