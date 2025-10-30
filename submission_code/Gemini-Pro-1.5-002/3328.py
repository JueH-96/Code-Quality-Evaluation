class Solution:
    def minOperations(self, k: int) -> int:
        if k <= 1:
            return 0
        
        ans = float('inf')
        for inc in range(1, int(k**0.5) + 2):
            curr = inc - 1
            rem = k - inc
            if rem <= 0:
                ans = min(ans, curr)
                continue
            
            curr += (rem + inc -1 ) // inc
            ans = min(ans, curr)
        return ans