class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = 10**9 + 7
        low, high = int(low), int(high)
        
        @lru_cache(None)
        def dfs(num, prev, is_limit, is_num):
            if num > high: return 0
            if num >= low and is_num: ans = 1
            else: ans = 0
            if num == 0: is_num = True
            if is_limit and num == high: is_limit = False
            if is_num:
                if prev == 0:
                    for d in [1, 9]:
                        ans += dfs(num*10+d, d, is_limit, is_num)
                else:
                    for d in [prev-1, prev+1]:
                        if 0 < d < 10:
                            ans += dfs(num*10+d, d, is_limit, is_num)
            else:
                for d in range(1, 10):
                    ans += dfs(num*10+d, d, is_limit, is_num)
            return ans % mod
        
        return dfs(0, 0, True, False)