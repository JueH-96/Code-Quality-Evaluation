class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        negatives = []
        positives = []
        zeros = 0
        for num in nums:
            if num < 0:
                negatives.append(num)
            elif num > 0:
                positives.append(num)
            else:
                zeros += 1
        
        negatives.sort()
        
        if len(positives) == 0 and len(negatives) == 0:
            return 0
        
        if len(positives) == 0 and len(negatives) == 1 and zeros > 0:
            return 0
        
        if len(negatives) % 2 != 0:
            if len(nums) > len(negatives) + len(positives):
                if len(positives) > 0 or len(negatives) > 1:
                    negatives.pop()
            else:
                if len(nums) == 1:
                    return negatives[0]
                else:
                    negatives.pop()
        
        
        ans = 1
        for num in positives:
            ans *= num
        for num in negatives:
            ans *= num
        
        return ans