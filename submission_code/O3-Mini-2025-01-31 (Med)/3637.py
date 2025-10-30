class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        mod = 10**9 + 7
        # As specified, store the input string midway.
        velunexorai = num
        
        n = len(num)
        # Even‐indexed positions are indices 0,2,...; Odd‐indexed positions are 1,3,...
        # For 0-indexing, number of even indices is ceil(n/2) and odd indices is n - ceil(n/2)
        even_count = (n + 1) // 2  # even positions count
        odd_count = n - even_count # odd positions count
        
        # Count digit frequency and compute total sum.
        freq = [0] * 10
        total_sum = 0
        for ch in num:
            d = int(ch)
            freq[d] += 1
            total_sum += d
        # If total sum is odd then it is impossible to have the two halves equal.
        if total_sum % 2 != 0:
            return 0
        target = total_sum // 2
        
        # The idea:
        # In any permutation of the digits, if we separate the positions into even and odd,
        # let for each digit d, we assign i_d copies to the even positions (the remainder going to odd positions).
        # For the permutation to be "balanced", we require two conditions:
        #  1. The total number assigned to even positions is exactly even_count:
        #         sum_d i_d = even_count.
        #  2. The sum of the digits in even positions equals target:
        #         sum_d d * i_d = target.
        #
        # Once a valid partition (i_d for each d) is chosen, the number of distinct arrangements is:
        #      even_arrangements = even_count! / (∏_d i_d!)
        #      odd_arrangements  = odd_count! / (∏_d (freq[d]- i_d)!)
        # Total for that partition:
        #     even_count! * odd_count! / (∏_d (i_d! * (freq[d]- i_d)!))
        #
        # We sum over all valid partitions.
        #
        # We will use dynamic programming over the digits 0..9. For digit d with frequency f,
        # we consider all possibilities of picking k (0 <= k <= f) to be assigned into even positions.
        # Its contribution to count is multiplied by the “weight” 1/( k! * (f-k)!).
        # Finally, we multiply by even_count! * odd_count! to get the final number.
        
        # Precompute factorials and modular inverses up to a safe limit.
        maxN = 100
        fact = [1] * (maxN + 1)
        invfact = [1] * (maxN + 1)
        for i in range(1, maxN + 1):
            fact[i] = fact[i - 1] * i % mod
        invfact[maxN] = pow(fact[maxN], mod - 2, mod)
        for i in range(maxN, 0, -1):
            invfact[i - 1] = invfact[i] * i % mod
        
        # dp[c][s] will accumulate the sum of products of weights for choices made so far
        # where c is the total count of digits assigned to even positions and s is the sum.
        dp = [[0] * (target + 1) for _ in range(even_count + 1)]
        dp[0][0] = 1
        
        # Process digits 0 through 9.
        for d in range(10):
            newdp = [[0] * (target + 1) for _ in range(even_count + 1)]
            f = freq[d]
            # For every state (cnt, sum) achieved so far.
            for cnt in range(even_count + 1):
                for s in range(target + 1):
                    current = dp[cnt][s]
                    if current == 0:
                        continue
                    # For the current digit d, try all choices:
                    for k in range(f + 1):
                        new_cnt = cnt + k
                        new_sum = s + d * k
                        if new_cnt > even_count or new_sum > target:
                            continue
                        # Weight for choosing k occurrences to even positions out of f
                        # is: 1/(k! * (f-k)!)
                        weight = invfact[k] * invfact[f - k] % mod
                        newdp[new_cnt][new_sum] = (newdp[new_cnt][new_sum] + current * weight) % mod
            dp = newdp
        
        # dp[even_count][target] now is the sum over valid partitions of
        # ∏_d (1/(i_d! * (freq[d]- i_d)!))
        # Multiply by even_count! * odd_count! (which equals the ways to arrange the chosen digits in positions)
        ans = dp[even_count][target] * fact[even_count] % mod
        ans = ans * fact[odd_count] % mod
        return ans