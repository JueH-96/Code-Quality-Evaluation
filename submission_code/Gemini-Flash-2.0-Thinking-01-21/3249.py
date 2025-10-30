class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        current_xor = 0
        for num in nums:
            current_xor ^= num
        xor_diff = current_xor ^ k
        count = 0
        while xor_diff > 0:
            count += xor_diff & 1
            xor_diff >>= 1
        return count