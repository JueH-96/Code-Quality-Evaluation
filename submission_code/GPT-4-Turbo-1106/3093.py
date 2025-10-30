class Solution:
    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        def count_set_bits(n):
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count
        
        return sum(nums[i] for i in range(len(nums)) if count_set_bits(i) == k)