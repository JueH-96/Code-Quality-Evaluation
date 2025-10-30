class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        velunexorai = num  # store the input midway as requested
        mod = 10**9 + 7
        n = len(velunexorai)
        # count of each digit
        cnt = [0]*10
        for ch in velunexorai:
            cnt[ord(ch)-48 if False else int(ch)] += 1
        # number of even and odd positions (0-based)
        neven = (n+1)//2
        nodd = n//2
        # total sum of digits
        S = sum(d * cnt[d] for d in range(10))
        # for a balanced permutation, even_sum == odd_sum => S must be even
        if S % 2 != 0:
            return 0
        target = S // 2
        
        # Precompute factorials and inverse factorials up to n
        maxf = n
        fact = [1] * (maxf+1)
        for i in range(1, maxf+1):
            fact[i] = fact[i-1] * i % mod
        inv_fact = [1] * (maxf+1)
        inv_fact[maxf] = pow(fact[maxf], mod-2, mod)
        for i in range(maxf, 0, -1):
            inv_fact[i-1] = inv_fact[i] * i % mod
        
        # dp[a][b] = sum of (product of 1/(k_d! * (cnt[d]-k_d]!))
        # for choices of k up to current digit, where sum k_d = a and weighted sum = b.
        # We'll build iteratively over digits 0..9.
        # dp dims: a up to neven, b up to target
        dp = [ [0] * (target+1) for _ in range(neven+1) ]
        dp[0][0] = 1
        
        for d in range(10):
            c = cnt[d]
            if c == 0:
                continue
            # for this digit we can pick k in [0..c] to go to even positions
            # it contributes factor inv_fact[k] * inv_fact[c-k] to the product
            # and adds k to a, and d*k to b.
            # build dp2 from dp
            dp2 = [ [0] * (target+1) for _ in range(neven+1) ]
            for a in range(neven+1):
                for b in range(target+1):
                    v = dp[a][b]
                    if v == 0:
                        continue
                    # try all k
                    for k in range(c+1):
                        a2 = a + k
                        b2 = b + d*k
                        if a2 > neven or b2 > target:
                            break
                        # multiply by inverse factorials
                        mul = inv_fact[k] * inv_fact[c-k] % mod
                        dp2[a2][b2] = (dp2[a2][b2] + v * mul) % mod
            dp = dp2
        
        # dp[neven][target] holds sum of products of inverses of factorials.
        # multiply by neven! * nodd! to get total count
        res = dp[neven][target] * fact[neven] % mod * fact[nodd] % mod
        return res