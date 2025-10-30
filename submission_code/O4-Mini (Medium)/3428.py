from typing import List

class Solution:
    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        seen = set()
        dup_xor = 0
        duplicates = set()
        
        for num in nums:
            if num in seen:
                # we only XOR each duplicate once
                if num not in duplicates:
                    duplicates.add(num)
                    dup_xor ^= num
            else:
                seen.add(num)
        
        return dup_xor