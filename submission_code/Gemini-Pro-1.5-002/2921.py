class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7
        ans = 0
        
        def isStepping(num_str):
            for i in range(len(num_str) - 1):
                if abs(int(num_str[i]) - int(num_str[i+1])) != 1:
                    return False
            return True

        def dfs(num_str):
            nonlocal ans
            if int(num_str) > int(high):
                return
            
            if int(num_str) >= int(low) and isStepping(num_str):
                ans = (ans + 1) % mod
            
            if len(num_str) >= len(high) and int(num_str) < int(high):
                return

            last_digit = int(num_str[-1])
            
            if last_digit > 0:
                dfs(num_str + str(last_digit - 1))
            if last_digit < 9:
                dfs(num_str + str(last_digit + 1))

        for i in range(1, 10):
            dfs(str(i))
        
        return ans