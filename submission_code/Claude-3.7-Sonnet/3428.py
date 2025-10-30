class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        seen_once = set()
        duplicates_xor = 0
        
        for num in nums:
            if num in seen_once:
                # We've seen this number before, so it appears twice
                seen_once.remove(num)
                duplicates_xor ^= num
            else:
                # First time seeing this number
                seen_once.add(num)
        
        return duplicates_xor