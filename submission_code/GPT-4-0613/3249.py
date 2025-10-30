class Solution:
    def minOperations(self, nums, k):
        xor = 0
        for num in nums:
            xor ^= num
        if xor == k:
            return 0
        else:
            return bin(xor ^ k).count('1')