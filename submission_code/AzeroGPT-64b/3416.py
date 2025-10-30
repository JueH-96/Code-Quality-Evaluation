class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        ans = 0
        lo=(len(str(min(nums))))
        
        for i in range(lo):
            m=collections.Counter(str(x)[i] for x in nums)
            for d1 in range(10):
                for d2 in range(d1+1,10):
                    l=m[str(d1)]*m[str(d2)]
                    ans+=l*2
        
        return ans