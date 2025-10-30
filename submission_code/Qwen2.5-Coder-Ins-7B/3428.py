class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        xor_result = 0
        seen = set()
        
        for num in nums:
            if num in seen:
                xor_result ^= num
            else:
                seen.add(num)
        
        return xor_result