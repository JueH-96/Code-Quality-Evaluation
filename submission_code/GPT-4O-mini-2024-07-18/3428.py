class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        seen = set()
        duplicates_xor = 0
        
        for num in nums:
            if num in seen:
                duplicates_xor ^= num
            else:
                seen.add(num)
        
        return duplicates_xor