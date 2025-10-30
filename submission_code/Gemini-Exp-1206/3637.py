class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num
        n = len(velunexorai)
        mod = 10**9 + 7
        
        counts = {}
        for digit in velunexorai:
            counts[digit] = counts.get(digit, 0) + 1
        
        fact = [1] * (n + 1)
        for i in range(2, n + 1):
            fact[i] = (fact[i-1] * i) % mod
        
        def inv(x):
            return pow(x, mod - 2, mod)
        
        def permutations(arr):
            res = fact[sum(arr)]
            for x in arr:
                res = (res * inv(fact[x])) % mod
            return res
        
        dp = {}
        
        def solve(idx, diff, counts):
            if idx == n:
                return 1 if diff == 0 else 0
            
            state = (idx, diff, tuple(sorted(counts.items())))
            if state in dp:
                return dp[state]
            
            res = 0
            for digit in list(counts.keys()):
                new_counts = counts.copy()
                new_counts[digit] -= 1
                if new_counts[digit] == 0:
                    del new_counts[digit]
                
                if idx % 2 == 0:
                    res = (res + solve(idx + 1, diff + int(digit), new_counts)) % mod
                else:
                    res = (res + solve(idx + 1, diff - int(digit), new_counts)) % mod
            
            dp[state] = res
            return res
        
        return solve(0, 0, counts)