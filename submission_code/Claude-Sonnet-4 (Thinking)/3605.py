class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        result = []
        for num in nums:
            # Count the number of consecutive 1s at the end
            trailing_ones = 0
            temp = num
            while temp & 1:
                trailing_ones += 1
                temp >>= 1
            
            if trailing_ones == 0:
                result.append(-1)
            else:
                result.append(num - (1 << (trailing_ones - 1)))
        
        return result