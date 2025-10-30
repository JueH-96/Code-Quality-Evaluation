class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7
        
        def dfs(i, zeros, ones, arr):
            if i == zero + one:
                return 1
            
            if zeros == 0 and ones == 1:
                for j in range(len(arr) - limit + 1):
                    if 0 not in arr[j:j+limit] or 1 not in arr[j:j+limit]:
                        return 0
                return 1
            
            total = 0
            if zeros > 0:
                total += dfs(i + 1, zeros - 1, ones, arr + [0])
            if ones > 0:
                total += dfs(i + 1, zeros, ones - 1, arr + [1])
            return total % MOD
        
        return dfs(0, zero, one, [])