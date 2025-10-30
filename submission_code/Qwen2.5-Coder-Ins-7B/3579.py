class Solution:
    def maxGoodNumber(self, nums: List[int]) -> int:
        nums.sort(reverse=True)
        return int(''.join(format(num, 'b') for num in nums), 2)