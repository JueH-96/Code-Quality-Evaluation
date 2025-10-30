class Solution:
    def minimizedStringLength(self, s: str) -> int:
        n = len(s)
        if n <= 2:
            return n
        
        dp = {}

        def solve(s):
            if s in dp:
                return dp[s]
            
            n = len(s)
            if n <= 1:
                return n
            
            ans = n
            for i in range(n):
                left = -1
                right = -1
                
                for j in range(i - 1, -1, -1):
                    if s[j] == s[i]:
                        left = j
                        break
                
                for j in range(i + 1, n):
                    if s[j] == s[i]:
                        right = j
                        break
                
                if left != -1 and right != -1:
                    new_s = s[:left] + s[left+1:right] + s[right+1:]
                    ans = min(ans, solve(new_s))
                elif left != -1:
                    new_s = s[:left] + s[left+1:]
                    ans = min(ans, solve(new_s))
                elif right != -1:
                    new_s = s[:right] + s[right+1:]
                    ans = min(ans, solve(new_s))
            
            dp[s] = ans
            return ans

        return solve(s)