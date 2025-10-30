class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        
        pos = []
        neg = []
        zeros = 0
        
        for num in nums:
            if num > 0:
                pos.append(num)
            elif num < 0:
                neg.append(num)
            else:
                zeros += 1
        
        if not pos and not neg:
            return 0
        
        neg.sort()
        
        if len(neg) % 2 != 0:
            if len(neg) == 1 and not pos and zeros > 0:
                return 0
            neg.pop()
        
        result = 1
        for num in pos + neg:
            result *= num
        
        return result