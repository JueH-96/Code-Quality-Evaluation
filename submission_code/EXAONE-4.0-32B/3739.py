class Solution:
    MOD = 10**9 + 7
    maxN = 100000
    fact = None
    inv_fact = None

    def distanceSum(self, m: int, n: int, k: int) -> int:
        total_cells = m * n
        if Solution.fact is None:
            Solution.fact = [1] * (Solution.maxN + 1)
            for i in range(1, Solution.maxN + 1):
                Solution.fact[i] = Solution.fact[i-1] * i % Solution.MOD
            Solution.inv_fact = [1] * (Solution.maxN + 1)
            Solution.inv_fact[Solution.maxN] = pow(Solution.fact[Solution.maxN], Solution.MOD - 2, Solution.MOD)
            for i in range(Solution.maxN, 0, -1):
                Solution.inv_fact[i-1] = Solution.inv_fact[i] * i % Solution.MOD
        
        inv6 = pow(6, Solution.MOD - 2, Solution.MOD)
        
        def T(x):
            if x < 2:
                return 0
            res = (x - 1) * x % Solution.MOD
            res = res * (x + 1) % Solution.MOD
            res = res * inv6 % Solution.MOD
            return res
        
        Tx = T(m)
        Ty = T(n)
        term_x = n * n % Solution.MOD * Tx % Solution.MOD
        term_y = m * m % Solution.MOD * Ty % Solution.MOD
        total_pair_contrib = (term_x + term_y) % Solution.MOD
        
        def nCr(n_val, r_val):
            if r_val < 0 or r_val > n_val:
                return 0
            return Solution.fact[n_val] * Solution.inv_fact[r_val] % Solution.MOD * Solution.inv_fact[n_val - r_val] % Solution.MOD
        
        n_arr = nCr(total_cells - 2, k - 2)
        ans = total_pair_contrib * n_arr % Solution.MOD
        return ans