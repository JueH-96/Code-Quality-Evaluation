class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        """
        Counts the number of positive integers x < n that are k-reducible.

        The method involves three main steps:
        1. Understanding the k-reducible property:
           An integer x is k-reducible if it can be reduced to 1 in at most k operations,
           where an operation is replacing x with its popcount.
           Let g(x) be the number of operations for x to become 1.
           g(1) = 0, and g(x) = 1 + g(popcount(x)) for x > 1.
           The condition for x to be k-reducible is g(x) <= k.
           This is equivalent to g(popcount(x)) <= k - 1 for any positive integer x.
           So, we need to count numbers x < n whose popcount p satisfies g(p) <= k - 1.

        2. Identifying target popcounts:
           We first find the set of "good" popcounts. These are integers p such that g(p) <= k - 1.
           We can compute g(p) for p from 1 to N (length of s) using memoized recursion.

        3. Counting numbers with target popcounts:
           The problem is now to count positive integers x < n such that popcount(x) is in our
           set of target popcounts. This is a classic digit DP problem that can be solved
           with a combinatorial approach. We iterate through the bits of the binary string s
           (representing n) from left to right. At each position i where s[i] is '1', we can
           choose to place a '0'. This makes the number guaranteed to be smaller than n.
           We then have the freedom to place the remaining required ones in the rest of the
           positions. The number of ways to do this can be calculated using combinations.
           We sum these counts for all valid positions and all target popcounts.
        """
        MOD = 10**9 + 7
        N = len(s)

        # Precompute factorials and their modular inverses for combinations C(n, k)
        fact = [1] * (N + 1)
        inv_fact = [1] * (N + 1)
        for i in range(1, N + 1):
            fact[i] = (fact[i - 1] * i) % MOD

        inv_fact[N] = pow(fact[N], MOD - 2, MOD)
        for i in range(N - 1, -1, -1):
            inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

        def combinations(n, k):
            if k < 0 or k > n:
                return 0
            numerator = fact[n]
            denominator = (inv_fact[k] * inv_fact[n - k]) % MOD
            return (numerator * denominator) % MOD

        # Memoization for g-values
        g_memo = {1: 0}
        def get_g(i):
            if i in g_memo:
                return g_memo[i]
            
            popcnt = bin(i).count('1')
            g_memo[i] = 1 + get_g(popcnt)
            return g_memo[i]

        if k == 0:
            return 0

        # Determine the set of popcounts that make a number k-reducible.
        target_popcounts = set()
        for p in range(1, N + 1):
            if get_g(p) <= k - 1:
                target_popcounts.add(p)

        if not target_popcounts:
            return 0

        # Count numbers x < n (represented by s) with a popcount in target_popcounts.
        # This method counts numbers in [0, n-1]. Since popcount(0)=0 is not in
        # target_popcounts, it effectively counts numbers in [1, n-1].
        total_ans = 0
        
        for p in target_popcounts:
            # Count numbers < n with popcount p
            count_for_p = 0
            ones_so_far = 0
            for i in range(N):
                rem_len = N - 1 - i
                if s[i] == '1':
                    # Place a '0' at position i. The number will be smaller than n.
                    # The prefix is s[0...i-1]0.
                    # We need to place p - ones_so_far more '1's in the remaining rem_len bits.
                    rem_ones = p - ones_so_far
                    count_for_p = (count_for_p + combinations(rem_len, rem_ones)) % MOD
                
                if s[i] == '1':
                    ones_so_far += 1
            
            total_ans = (total_ans + count_for_p) % MOD
            
        return total_ans