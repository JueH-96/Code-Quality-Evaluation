class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        initial_xor = 0
        for num in nums:
            initial_xor ^= num

        diff = initial_xor ^ k
        operations = 0
        while diff > 0:
            if diff & 1:
                operations += 1
            diff >>= 1
        return operations