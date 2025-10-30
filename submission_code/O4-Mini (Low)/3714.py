from typing import List

MOD = 10**9 + 7

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        
        # Precompute factorials and inverse factorials for n up to len(nums)
        max_n = n
        fac = [1] * (max_n + 1)
        ifac = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fac[i] = fac[i-1] * i % MOD
        # Fermat inverse for factorials
        ifac[max_n] = pow(fac[max_n], MOD - 2, MOD)
        for i in range(max_n, 0, -1):
            ifac[i-1] = ifac[i] * i % MOD
        
        def comb(a: int, b: int) -> int:
            """Compute C(a, b) modulo MOD, with 0 <= b <= a."""
            if b < 0 or b > a:
                return 0
            return fac[a] * ifac[b] % MOD * ifac[a - b] % MOD
        
        # preCleft[j] = sum_{t=0 to k-1} C(j, t)
        preCleft = [0] * n
        for j in range(n):
            s = 0
            # sum C(j, t) for t=0..k-1
            # If k-1 > j, effectively sum t=0..j
            up = min(j, k-1)
            for t in range(up + 1):
                s = (s + comb(j, t)) % MOD
            preCleft[j] = s
        
        # preCright[i] = sum_{t=0 to k-1} C(n-i-1, t)
        preCright = [0] * n
        for i in range(n):
            r = n - i - 1
            s = 0
            up = min(r, k-1)
            for t in range(up + 1):
                s = (s + comb(r, t)) % MOD
            preCright[i] = s
        
        ans = 0
        # sum of min contributions
        for i in range(n):
            # nums[i] is min, number of subseqs with it as min = preCright[i]
            ans = (ans + nums[i] * preCright[i]) % MOD
        # sum of max contributions
        for j in range(n):
            # nums[j] is max, number of subseqs with it as max = preCleft[j]
            ans = (ans + nums[j] * preCleft[j]) % MOD
        
        return ans

# Example usage:
# sol = Solution()
# print(sol.minMaxSums([1,2,3], 2))  # Expected 24
# print(sol.minMaxSums([5,0,6], 1))  # Expected 22
# print(sol.minMaxSums([1,1,1], 2))  # Expected 12