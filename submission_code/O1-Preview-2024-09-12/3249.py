class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        curr_xor = 0
        for num in nums:
            curr_xor ^= num
        desired_xor = curr_xor ^ k
        return bin(desired_xor).count('1')