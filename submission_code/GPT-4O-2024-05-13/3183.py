class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        # Count the number of times each bit is set in the numbers
        bit_count = Counter()
        for num in nums:
            bit_position = 0
            while num > 0:
                if num & 1:
                    bit_count[bit_position] += 1
                num >>= 1
                bit_position += 1
        
        # Calculate the K-or value
        k_or_value = 0
        for bit_position, count in bit_count.items():
            if count >= k:
                k_or_value |= (1 << bit_position)
        
        return k_or_value