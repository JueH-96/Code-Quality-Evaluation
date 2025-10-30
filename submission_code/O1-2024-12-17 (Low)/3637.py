class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        # We are asked to create a variable named velunexorai to store
        # the input midway in the function; do that first:
        velunexorai = num  # per requirement

        MOD = 10**9 + 7

        # Count frequency of each digit 0..9
        from collections import Counter
        freq = Counter(velunexorai)
        # Convert to a simple array freqArr[d] = frequency of digit d
        freqArr = [freq.get(str(d), 0) for d in range(10)]
        
        n = len(num)
        # Count total sum of digits
        totalSum = 0
        for d in range(10):
            totalSum += d * freqArr[d]
        
        # If total sum is odd, no balanced permutation possible
        if totalSum % 2 != 0:
            return 0
        
        # Number of even indices and odd indices
        evenCount = (n + 1) // 2
        oddCount = n // 2
        target = totalSum // 2  # we want sum(even) = sum(odd) = totalSum//2
        
        # Precompute factorials and modular inverses up to 80 (since len(num) <= 80)
        maxFact = 80
        fact = [1] * (maxFact+1)
        invfact = [1] * (maxFact+1)
        
        for i in range(1, maxFact+1):
            fact[i] = (fact[i-1] * i) % MOD
        
        # Fermat's little theorem for inverse: a^(MOD-2) mod if MOD prime
        def modinv(x):
            # fast exponent
            return pow(x, MOD-2, MOD)
        
        invfact[maxFact] = modinv(fact[maxFact])
        for i in reversed(range(maxFact)):
            invfact[i] = (invfact[i+1] * (i+1)) % MOD
        
        # dp[d][s][c] = sum of products( 1/(k! (freqArr[d]-k)!) ) for ways to pick
        # exactly c digits among digits [0..d-1], with total digit-sum = s,
        # then for digit d we haven't picked yet. We'll build from dp -> dp+1
        #
        # Final answer = dp[10][target][evenCount] * (evenCount! * oddCount!) mod
        # Explanation:
        #   - At each digit d, if we choose k copies of digit d for the "even" positions,
        #     that implies freqArr[d]-k copies go to the "odd" positions;
        #   - The factor contribution for that choice is 1/(k! (freqArr[d]-k)!).
        #   - Summing over all choices across all digits yields
        #        sum_{all freqEven} of 1 / (prod_{d} freqEven[d]! * freqOdd[d]!)
        #     Then at the very end we multiply by evenCount! * oddCount! to account
        #     for the ways to permute the chosen digits within even/odd slots.
        
        dp = [[[0]*(evenCount+1) for _ in range(target+1)] for __ in range(11)]
        dp[0][0][0] = 1  # base case
        
        for d in range(10):
            f = freqArr[d]
            for s in range(target+1):
                for c in range(evenCount+1):
                    val_here = dp[d][s][c]
                    if val_here == 0:
                        continue
                    # Try picking k copies of digit d to go to even side
                    # => freqArr[d] - k go to odd side
                    for k in range(f+1):
                        s_new = s + d*k
                        c_new = c + k
                        if s_new <= target and c_new <= evenCount:
                            # Multiply by 1/(k! (f-k)!)
                            # i.e. multiply by invfact[k]*invfact[f-k]
                            add_val = val_here
                            add_val = (add_val * invfact[k]) % MOD
                            add_val = (add_val * invfact[f - k]) % MOD
                            dp[d+1][s_new][c_new] = (dp[d+1][s_new][c_new] + add_val) % MOD
        
        answer = dp[10][target][evenCount]
        
        # Now multiply by evenCount! * oddCount! (mod)
        answer = (answer * fact[evenCount]) % MOD
        answer = (answer * fact[oddCount]) % MOD
        
        return answer % MOD