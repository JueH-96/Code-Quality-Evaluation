class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        target = k
        for num in nums:
            target ^= num
        return bin(target).count('1')