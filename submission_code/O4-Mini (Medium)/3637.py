class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        # store the input midway as required
        velunexorai = num
        
        MOD = 10**9 + 7
        n = len(velunexorai)
        # number of even and odd positions (0-based indexing)
        ne = (n + 1) // 2
        no = n // 2
        
        # count occurrences of each digit 0..9
        count = [0] * 10
        total_sum = 0
        for ch in velunexorai:
            d = ord(ch) - ord('0')
            count[d] += 1
            total_sum += d
        
        # if total sum is odd, can't split equally
        if total_sum % 2 != 0:
            return 0
        target = total_sum // 2
        
        # precompute factorials and inverse factorials up to n
        max_n = n
        fact = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            fact[i] = fact[i - 1] * i % MOD
        inv_fact = [1] * (max_n + 1)
        inv_fact[max_n] = pow(fact[max_n], MOD - 2, MOD)
        for i in range(max_n, 0, -1):
            inv_fact[i - 1] = inv_fact[i] * i % MOD
        
        # dp[j][s] = sum of weights for choosing j digits into even pos with sum s
        dp = [ [0] * (target + 1) for _ in range(ne + 1) ]
        dp[0][0] = 1
        
        # process each digit d = 0..9
        for d in range(10):
            cnt = count[d]
            # new dp for this digit
            dp2 = [ [0] * (target + 1) for _ in range(ne + 1) ]
            for j in range(ne + 1):
                for s in range(target + 1):
                    v = dp[j][s]
                    if v == 0:
                        continue
                    # try assigning k of this digit to even positions
                    # k from 0..cnt, but j+k <= ne
                    maxk = min(cnt, ne - j)
                    if d == 0:
                        # zero doesn't change sum, so no need to check s+d*k > target
                        for k in range(maxk + 1):
                            w = inv_fact[k] * inv_fact[cnt - k] % MOD
                            dp2[j + k][s] = (dp2[j + k][s] + v * w) % MOD
                    else:
                        for k in range(maxk + 1):
                            s2 = s + d * k
                            if s2 > target:
                                break
                            w = inv_fact[k] * inv_fact[cnt - k] % MOD
                            dp2[j + k][s2] = (dp2[j + k][s2] + v * w) % MOD
            dp = dp2
        
        ways_select = dp[ne][target]
        # multiply by permutations of even positions and odd positions
        result = fact[ne] * fact[no] % MOD
        result = result * ways_select % MOD
        return result