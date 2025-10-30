class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        max_bit = max(max(nums, default=0), k).bit_length()
        c = [0] * max_bit
        for num in nums:
            for i in range(max_bit):
                if num & (1 << i):
                    c[i] += 1
        operations = 0
        for i in range(max_bit):
            desired_bit = (k >> i) & 1
            current_bit = c[i] % 2
            if current_bit != desired_bit:
                operations += 1
        return operations