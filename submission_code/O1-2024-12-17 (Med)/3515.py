class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        sumSingle = sum(x for x in nums if x < 10)
        sumDouble = sum(x for x in nums if x >= 10)
        return sumSingle > sumDouble or sumDouble > sumSingle