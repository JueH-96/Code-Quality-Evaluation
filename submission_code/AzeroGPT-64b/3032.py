class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        dp = [0] * n
        for _ in range(max(50, k + 1)):
            for i in range(n):
                dp[receiver[i]] += i + 1
            k -= 1
            if k == 0:
                break
        
        if k > 0:
            pow2 = []
            x = k
            while x > 0:
                pow2.append(x)
                x //= 2
            
            ans = 0
            add = 0
            for i in range(n):
                value = dp[i]
                x = i
                for p in pow2:
                    if p <= k:
                        value += dp[x]
                        k -= p
                        x = receiver[x]
                if k > 0:
                    value += k * x
                k = pow2[-1]
                ans = max(ans, value)
            return ans
        else:
            return max(dp)