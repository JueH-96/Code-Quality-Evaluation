class Solution:
    def maxStrength(self, nums: List[int]) -> int:
        n = len(nums)
        pos = [x for x in nums if x > 0]
        neg = [x for x in nums if x < 0]
        zero = 0 in nums
        
        if len(pos) == 0 and len(neg) <= 1 and zero == False:
            return max(nums)
        
        if len(neg) % 2 == 0:
            neg_product = reduce(lambda x, y: x*y, neg)
        else:
            neg.sort()
            neg_product = reduce(lambda x, y: x*y, neg[:-1])
        
        pos_product = reduce(lambda x, y: x*y, pos)
        
        return neg_product * pos_product