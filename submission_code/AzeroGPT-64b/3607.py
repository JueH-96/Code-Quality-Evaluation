from math import gcd
from functools import reduce
class Solution:
    def minOperations(self, num: List[int]) -> int:
        count = [0]*(len(num)-1)
        bef =  num[0]
        for i in range(1,len(num)):
            if num[i]<bef:
                count[i-1]=gcd(bef,num[i])
            bef = num[i]
        
        arr = reduce(lambda a, b:[a[i]+b[i] if b[i] else a[i]  for i in range(len(a))] , ([0]+count for _ in range(len(num))))
        m = min(arr[1:])
        if m ==0:
            return -1
        res  = 0
        bef  = num[0]
        for i in range(1,len(num)):
            if bef>num[i]:
                now = num[i]
                num[i]=bef
                bef = max(bef,num[i]*m//gcd(bef,num[i]))
                res += 1
            else: bef = num[i]
        if bef<num[len(num)-1]:
            return -1
        return res