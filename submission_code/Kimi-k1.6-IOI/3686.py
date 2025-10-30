class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            return 0
        
        base = 911382629
        mod = 10**18 + 3
        
        power = [1] * (n + 1)  # power[i] = base^i mod mod
        for i in range(1, n + 1):
            power[i] = (power[i-1] * base) % mod
        
        prefix_hash = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_hash[i] = (prefix_hash[i-1] * base + nums[i-1]) % mod
        
        def get_hash(l, r):
            if l > r:
                return 0
            length = r - l + 1
            res = (prefix_hash[r+1] - (prefix_hash[l] * power[length]) % mod) % mod
            return res if res >= 0 else res + mod
        
        count = 0
        
        for i in range(1, n - 1):  # i ranges from 1 to n-2 inclusive
            for j in range(i + 1, n):  # j ranges from i+1 to n-1 inclusive
                condition1 = False
                if j >= 2 * i:
                    end = 2 * i - 1
                    if end < n:
                        h1 = get_hash(0, i - 1)
                        h2 = get_hash(i, end)
                        if h1 == h2:
                            condition1 = True
                
                condition2 = False
                max_j_cond2 = (n + i) // 2
                if j <= max_j_cond2:
                    b = j - i
                    end3 = j + b - 1
                    if end3 < n:
                        h3 = get_hash(i, j - 1)
                        h4 = get_hash(j, end3)
                        if h3 == h4:
                            condition2 = True
                
                if condition1 or condition2:
                    count += 1
        
        return count