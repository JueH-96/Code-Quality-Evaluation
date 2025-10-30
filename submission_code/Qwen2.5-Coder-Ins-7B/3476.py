class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        remainder_1 = sum(1 for num in nums if num % 3 == 1)
        remainder_2 = sum(1 for num in nums if num % 3 == 2)
        
        if remainder_1 == remainder_2:
            return remainder_1
        elif remainder_1 > remainder_2:
            return remainder_2 + (remainder_1 - remainder_2) // 3
        else:
            return remainder_1 + (remainder_2 - remainder_1) // 3