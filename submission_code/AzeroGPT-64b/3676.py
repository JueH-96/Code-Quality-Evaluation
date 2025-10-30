class Solution:
    def smallestNumber(self, n: int) -> int:
        x=1
        while n>0:
            x<<=1
            n-=1
        return x-1