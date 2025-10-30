class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        nums.sort(key=lambda x: bin(x)[::-1])
        return int(''.join(bin(num)[2:] for num in nums), 2)