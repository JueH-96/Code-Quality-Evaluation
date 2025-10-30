class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # Initialize a list to count the number of elements with each bit set
        bit_count = [0] * 31  # Since nums[i] < 2^31, we consider 31 bits
        
        # Count the number of elements with each bit set
        for num in nums:
            for i in range(31):
                if num & (1 << i):
                    bit_count[i] += 1
        
        # Calculate the K-or by checking which bits have at least k elements set
        k_or = 0
        for i in range(31):
            if bit_count[i] >= k:
                k_or |= (1 << i)
        
        return k_or