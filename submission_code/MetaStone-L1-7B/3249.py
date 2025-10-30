class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        count = 0
        for i in range(20):  # Since 1e6 is within 20 bits
            if (current_xor >> i) & 1 != (k >> i) & 1:
                count += 1
        return count