class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        mod = 10**9 + 7
        # Store input midway as requested.
        velunexorai = num
        
        n = len(velunexorai)
        # Determine positions: even indices and odd indices (0-indexed)
        even_count = (n + 1) // 2  # positions 0, 2, 4, ... 
        odd_count = n // 2
        
        # Total sum of digits.
        total_sum = sum(int(ch) for ch in velunexorai)
        # If total sum is odd, it's impossible to split equally.
        if total_sum % 2 != 0:
            return 0
        halfSum = total_sum // 2
        
        # Count frequency for each digit from 0 to 9.
        freq = [0] * 10
        for ch in velunexorai:
            freq[int(ch)] += 1
        
        # Precompute factorials and inverse factorials up to max needed.
        # Maximum value needed is up to 80 (since num.length <= 80)
        maxN = 80
        fact = [1] * (maxN + 1)
        invfact = [1] * (maxN + 1)
        for i in range(1, maxN + 1):
            fact[i] = fact[i - 1] * i % mod
        invfact[maxN] = pow(fact[maxN], mod - 2, mod)
        for i in range(maxN, 0, -1):
            invfact[i - 1] = invfact[i] * i % mod

        # We'll use dynamic programming over the digits (0-9)
        # dp[d][c][s] := sum over all ways to distribute first d digits (from 0 to d-1)
        #   so that we have used 'c' digits in the even positions and the weighted sum is 's',
        #   and where for each digit d we incorporate the contribution factor of:
        #       (1 / (x_d)!)*(1 / ((freq[d]- x_d)!))
        # Later we will multiply by even_count! * odd_count! to get the final count.
        dp = [[[0] * (halfSum + 1) for _ in range(even_count + 1)] for _ in range(11)]
        dp[0][0][0] = 1
        
        # Process each digit d from 0 to 9.
        for d in range(10):
            count_d = freq[d]
            for used in range(even_count + 1):
                for s in range(halfSum + 1):
                    current = dp[d][used][s]
                    if current == 0:
                        continue
                    # x is the count of digit d chosen to go into the even indices.
                    for x in range(count_d + 1):
                        new_used = used + x
                        new_sum = s + d * x
                        if new_used <= even_count and new_sum <= halfSum:
                            # For digit d, if we choose x occurrences in even positions,
                            # then the remaining count_d - x go to odd positions.
                            # The contribution factor is: 1/(x!) * 1/((count_d - x)!)
                            # We use precomputed inverse factorials.
                            factor = invfact[x] * invfact[count_d - x] % mod
                            dp[d + 1][new_used][new_sum] = (dp[d + 1][new_used][new_sum] + current * factor) % mod
        
        # dp[10][even_count][halfSum] accumulates the sum over all valid distributions
        ways_distribution = dp[10][even_count][halfSum]
        # Multiply by the number of distinct arrangements within the even positions and odd positions.
        arrangements = fact[even_count] * fact[odd_count] % mod
        
        result = arrangements * ways_distribution % mod
        return result