class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        max_bit = max(max(nums).bit_length(), k.bit_length()) - 1
        fn = lambda x: int(x) ^ int(k >> pos & 1)
        return sum(min(sum(((n >> pos) & 1 for n in nums), 0), fn(sum(((n >> pos) & 1) for n in nums), 0)) 
                   for pos in range(max_bit, -1, -1))