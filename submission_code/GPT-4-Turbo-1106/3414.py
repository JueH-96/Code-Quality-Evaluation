class Solution:
    def waysToReachStair(self, k: int) -> int:
        MOD = 10**9 + 7
        
        # If k is 0, there are two ways to stay at 0 (either do nothing or go up and down)
        if k == 0:
            return 2
        
        # If k is odd, we can only reach it by going up, which is equivalent to k-1
        if k % 2 == 1:
            return self.waysToReachStair(k - 1)
        
        # If k is even, we can reach it by going up to k/2 and then doubling the jump
        return (2 * self.waysToReachStair(k // 2)) % MOD