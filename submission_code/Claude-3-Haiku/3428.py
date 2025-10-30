class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result ^= num
        
        for i in range(1, len(nums)+1):
            result ^= i
        
        return result