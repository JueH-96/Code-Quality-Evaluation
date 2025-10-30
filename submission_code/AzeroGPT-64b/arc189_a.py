MOD=998244353
from itertools import accumulate
class Solution:
    def numberOfOperations(self, A):
        n=len(A)
        A=[(x,i) for i,x in enumerate(A) if x]
        presum=[0]+list(accumulate([a[1] for a in A], max))
        tot=0
        for i,(x,i0) in enumerate(A):
            if i!=0 and x==A[i-1][0]:
                tot+=presum[i]-A[i-1][1]-1
            if i!=len(A)-1 and x==A[i+1][0]:
                tot+=A[i+1][1]-presum[i]-1
        return pow(3,tot,MOD)*pow(2,~(len(A)-1),MOD)%MOD