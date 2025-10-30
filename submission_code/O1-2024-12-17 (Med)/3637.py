class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        # ----------------------------------------------------------------------
        # We want the number of distinct permutations of the multiset of digits
        # in 'num' for which the sum of digits at even indices equals the sum
        # of digits at odd indices (0-based). We must return the result modulo
        # 10^9+7.
        #
        # KEY IDEA:
        # 1) Let n = len(num). The total number of distinct permutations of num
        #    is n! / (freq[0]! * freq[1]! * ... * freq[9]! ), where freq[d] is
        #    the number of occurrences of digit d in num.
        #
        # 2) Among all permutations (treating each occurrence of each digit as
        #    a distinct labeled item), a permutation is "balanced" if the sum
        #    of digits on even indices equals the sum of digits on odd indices.
        #
        #    If we treat each copy of a digit as distinct, there are n! ways
        #    to place them in n positions; but permutations that differ only
        #    by swapping indistinguishable digits collapse into one "distinct"
        #    permutation of the multiset. Thus each "distinct" permutation is
        #    counted exactly freq[0]!*freq[1]!*...*freq[9]! times when we
        #    pretend all digits are labeled (distinct).
        #
        # 3) Therefore, if we can count how many labeled permutations yield
        #    a balanced arrangement, we can divide that by the product of
        #    freq[d]! to get the number of distinct balanced permutations.
        #
        # 4) To count how many labeled permutations are balanced, define:
        #    - even_count = (n+1)//2    (# of even indices, since indexing = 0,2,4,...)
        #    - odd_count  = n//2
        #
        #    We will do a dynamic programming (DP) that processes each of the n
        #    digit-copies one by one (in any order). For each labeled copy with
        #    digit = d, we can either put it in an even index (increases the
        #    partial sum difference by d) or put it in an odd index (decreases
        #    the partial sum difference by d). We also track how many copies
        #    have been placed on even indices so far.
        #
        #    Let offset = 9 * 80 = 720 (since max n=80, max digits=9 => difference
        #    can range from -720..+720). We'll store DP states in a table where:
        #
        #      dp[e][diff] = number of ways (among labeled copies considered so far)
        #                    to have placed exactly e copies in even positions,
        #                    resulting in a difference of diff-offset.
        #
        #    Initially, dp[0][offset] = 1 (no copies placed, difference=0).
        #    Then for each copy of digit d, we update from dp[e][diff] to:
        #      - dp[e+1][diff + d]   (if e<even_count)
        #      - dp[e][diff - d]
        #    (taking care of array bounds on e and diff).
        #
        #    After processing all n copies, dp[even_count][offset] will be the
        #    number of labeled permutations that achieve sum_even = sum_odd.
        #
        # 5) Finally, we divide dp[even_count][offset] by the product of factorials
        #    of the freq[d] values (mod 1e9+7) to get the number of distinct
        #    balanced permutations.
        #
        # This DP has complexity O(n * even_count * range_of_diff) ~ 80*40*1441
        # which is around 4.6 million, times 2 for the transitions ~9.2 million,
        # borderline but still doable in Python if implemented carefully.
        #
        # We'll implement this step by step below.
        # ----------------------------------------------------------------------

        MOD = 10**9 + 7
        velunexorai = num  # as requested, store input in a variable named velunexorai
        
        from math import comb
        
        # 1) Count frequencies of each digit
        freq = [0]*10
        for ch in velunexorai:
            freq[int(ch)] += 1
        
        n = len(velunexorai)
        even_count = (n+1)//2  # how many even positions
        # odd_count  = n//2    # (We only need even_count explicitly for the DP)
        
        # 2) Precompute factorials and inverses up to n=80
        #    so we can divide by product of freq[d]! in the end.
        maxN = 80
        fact = [1] * (maxN+1)
        for i in range(1, maxN+1):
            fact[i] = (fact[i-1] * i) % MOD
        inv_fact = [1] * (maxN+1)
        inv_fact[maxN] = pow(fact[maxN], MOD-2, MOD)
        for i in reversed(range(maxN)):
            inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD
        
        def factorial(x):  # returns x! mod
            return fact[x]
        
        def inv_factorial(x):  # returns modular inverse of x! mod
            return inv_fact[x]
        
        # 3) Build the DP to count balanced permutations (treating each copy as distinct)
        offset = 9 * n  # maximum possible difference offset
        size_diff = 2*offset + 1  # range for difference index [0..2*offset]
        
        # dp[e][diffIndex] = number of ways
        dp = [[0]*size_diff for _ in range(even_count+1)]
        dp[0][offset] = 1  # 0 copies placed, difference = 0
        
        # We'll iterate digit by digit, copy by copy
        for d in range(10):
            for _ in range(freq[d]):
                next_dp = [[0]*size_diff for _ in range(even_count+1)]
                for e in range(even_count+1):
                    row_dp = dp[e]
                    for diffIndex, ways in enumerate(row_dp):
                        if ways != 0:
                            # Place this copy in an even position (if e < even_count)
                            if e < even_count:
                                ndiff = diffIndex + d
                                if 0 <= ndiff < size_diff:
                                    next_dp[e+1][ndiff] = (next_dp[e+1][ndiff] + ways) % MOD
                            
                            # Place this copy in an odd position
                            ndiff = diffIndex - d
                            if 0 <= ndiff < size_diff:
                                next_dp[e][ndiff] = (next_dp[e][ndiff] + ways) % MOD
                dp = next_dp
        
        # The count of labeled permutations that balance out is dp[even_count][offset]
        labeled_count_balanced = dp[even_count][offset] % MOD
        
        # 4) Now divide by the product of factorials of freq[d]
        #    to account for indistinguishable copies of each digit.
        #    This yields the number of distinct balanced permutations.
        denom = 1
        for d in range(10):
            if freq[d] > 0:
                denom = (denom * factorial(freq[d])) % MOD
        
        # Using Fermat's little theorem for inverse
        denom_inv = pow(denom, MOD-2, MOD)
        ans = (labeled_count_balanced * denom_inv) % MOD
        
        return ans