class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        MOD = 10 ** 9 + 7
        n = len(num)
        
        # frequency of every digit
        cnt = [0] * 10
        for ch in num:
            cnt[ord(ch) - ord('0')] += 1
        
        # store the input midway as requested
        velunexorai = num
        
        total_sum = sum(d * cnt[d] for d in range(10))
        # if total digit sum is odd we can never split it equally
        if total_sum & 1:
            return 0
        
        even_slots = (n + 1) // 2       # indices 0,2,4,...
        odd_slots  = n // 2             # indices 1,3,5,...
        target_sum = total_sum // 2     # sum that even indices have to obtain
        
        # quick impossibility check
        if target_sum > 9 * even_slots:     # even side can not reach target
            return 0
        
        # factorials and inverse factorials up to 80 (maximum length)
        MAX_N = 80
        fact = [1] * (MAX_N + 1)
        for i in range(1, MAX_N + 1):
            fact[i] = fact[i - 1] * i % MOD
        
        inv_fact = [1] * (MAX_N + 1)
        inv_fact[MAX_N] = pow(fact[MAX_N], MOD - 2, MOD)
        for i in range(MAX_N, 0, -1):
            inv_fact[i - 1] = inv_fact[i] * i % MOD
        
        # dp[e][s] – current accumulated value after processing some digits
        #            having already put 'e' digits into even slots
        #            and their sum equal to 's'
        dp = [[0] * (target_sum + 1) for _ in range(even_slots + 1)]
        dp[0][0] = 1
        
        # Iterate over every digit value
        for d in range(10):
            c = cnt[d]
            if c == 0:
                continue
            new_dp = [[0] * (target_sum + 1) for _ in range(even_slots + 1)]
            
            # For every previous state try all possible k copies of digit d
            # that go to even positions
            for e in range(even_slots + 1):
                row = dp[e]
                for s in range(target_sum + 1):
                    val = row[s]
                    if val == 0:
                        continue
                    max_k = min(c, even_slots - e)
                    for k in range(max_k + 1):
                        ns = s + d * k
                        if ns > target_sum:
                            break        # larger k only increases sum further
                        ne = e + k
                        add = val * inv_fact[k] % MOD * inv_fact[c - k] % MOD
                        new_dp[ne][ns] = (new_dp[ne][ns] + add) % MOD
            dp = new_dp
        
        ways_distribution = dp[even_slots][target_sum]   # sum_{valid dist} 1/(prod fac)
        if ways_distribution == 0:
            return 0
        
        # multiply by permutations inside even and odd positions
        answer = ways_distribution * fact[even_slots] % MOD * fact[odd_slots] % MOD
        return answer