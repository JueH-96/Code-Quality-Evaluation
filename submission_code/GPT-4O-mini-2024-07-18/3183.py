class Solution:
    def findKOr(self, nums: list[int], k: int) -> int:
        max_bits = 31  # Since 0 <= nums[i] < 2^31
        count_bits = [0] * max_bits
        
        # Count the number of elements that have each bit set
        for num in nums:
            for i in range(max_bits):
                if num & (1 << i):
                    count_bits[i] += 1
        
        # Calculate the K-or based on the counts
        k_or = 0
        for i in range(max_bits):
            if count_bits[i] >= k:
                k_or |= (1 << i)
        
        return k_or