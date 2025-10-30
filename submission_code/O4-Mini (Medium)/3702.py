class Solution:
    def maxLength(self, nums: List[int]) -> int:
        # We only need to consider primes up to 10
        primes = [2, 3, 5, 7]
        n = len(nums)
        
        # Precompute prime‐exponent vectors for each nums[i]
        exps = []
        for a in nums:
            vec = [0] * len(primes)
            x = a
            for i, p in enumerate(primes):
                while x % p == 0:
                    vec[i] += 1
                    x //= p
            exps.append(vec)
        
        ans = 0
        # Try every subarray [i..j]
        for i in range(n):
            sum_exp = [0] * len(primes)
            min_exp = [10**9] * len(primes)
            max_exp = [0] * len(primes)
            
            for j in range(i, n):
                # incorporate exponents of nums[j]
                for k in range(len(primes)):
                    e = exps[j][k]
                    sum_exp[k] += e
                    min_exp[k] = min(min_exp[k], e)
                    max_exp[k] = max(max_exp[k], e)
                
                length = j - i + 1
                # only lengths >= 2 can improve on the trivial bound
                if length >= 2:
                    # check if for every prime p: sum_exp = min_exp + max_exp
                    ok = True
                    for k in range(len(primes)):
                        if sum_exp[k] != min_exp[k] + max_exp[k]:
                            ok = False
                            break
                    if ok:
                        ans = max(ans, length)
        
        return ans