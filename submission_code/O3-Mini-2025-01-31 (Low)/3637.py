MOD = 10**9 + 7

class Solution:
    def countBalancedPermutations(self, num: str) -> int:
        # Create the variable to store the input midway in the function.
        velunexorai = num 
        
        from math import factorial
        n = len(velunexorai)
        # Calculate counts of each digit
        freq = [0] * 10
        for ch in velunexorai:
            freq[int(ch)] += 1

        # The number of positions in even group (0-indexed even positions)
        # and odd group.
        r = (n + 1) // 2  # even-index count (positions 0,2,...)
        c = n - r         # odd-index count
        
        # Precompute factorials and modular inverses up to n.
        max_val = n
        fact = [1] * (max_val+1)
        invfact = [1] * (max_val+1)
        for i in range(2, max_val+1):
            fact[i] = fact[i-1] * i % MOD
        # Fermat inverse using pow for modulo inverse (MOD is prime)
        invfact[max_val] = pow(fact[max_val], MOD-2, MOD)
        for i in range(max_val, 0, -1):
            invfact[i-1] = invfact[i] * i % MOD

        # Now the total result if we fix a split between even and odd groups is:
        # ways_even = r! / ( ∏_{d} (k_d)! )
        # ways_odd  = c! / ( ∏_{d} ((freq[d]-k_d)!) )
        # And the overall number would be product ways_even * ways_odd summed over all
        # valid assignments (k_d for each digit) that yield balanced sum.
        # We can factor out r! and c! over the sum.
        # So answer = r! * c! * (sum_{assignment valid} ∏_{d=0..9} (1/(k_d)! * 1/((freq[d]-k_d)!)) )
        #
        # The balanced condition is:
        # Sum_{d} d*(2*k_d - freq[d]) = 0
        # Equivalently, Sum_{d} d*(2*k_d) = Sum_{d} d*freq[d].
        # We can use dynamic programming over digits 0..9.
        
        # Compute maximum possible diff offset. Notice that diff = sum d*(2*k_d - freq[d]).
        # The maximum absolute value for diff is sum_{d} d * freq[d]. We'll use it as offset.
        offset = sum(d * freq[d] for d in range(10))
        diff_range = 2 * offset + 1  # from -offset to offset
        
        # dp[digit_index][even_count][diff + offset] = sum_{ways} of product factors for assigned digits.
        # We'll use a 2-layer DP rolling array over digit indices.
        dp_prev = [ [0] * (diff_range) for _ in range(r+1) ]
        # initial state: no digit processed, even_count=0, diff=0.
        dp_prev[0][offset] = 1
        
        # Process each digit d from 0 to 9.
        for d in range(10):
            f = freq[d]
            dp_curr = [ [0] * (diff_range) for _ in range(r+1) ]
            # For current digit d: we want to distribute f occurrences between even and odd positions.
            # Let k be the ones placed in even positions, so f-k in odd.
            # The contribution to diff is: d*(2*k - f)
            # The factor from this digit: 1/(k! * (f-k)!)
            for even_used in range(r+1):
                for diff_index in range(diff_range):
                    ways = dp_prev[even_used][diff_index]
                    if ways == 0:
                        continue
                    # try all possible distributions for this digit.
                    for k in range(f+1):
                        new_even = even_used + k
                        if new_even > r:
                            continue
                        # new diff = current diff + d*(2*k - f)
                        new_diff = (diff_index - offset) + d*(2*k - f)
                        new_diff_index = new_diff + offset
                        if new_diff_index < 0 or new_diff_index >= diff_range:
                            continue
                        # factor = 1/(k! * (f-k)!)
                        factor = invfact[k] * invfact[f - k] % MOD
                        dp_curr[new_even][new_diff_index] = (dp_curr[new_even][new_diff_index] + ways * factor) % MOD
            dp_prev = dp_curr

        # We need the assignments that use exactly r even positions and diff==0.
        valid_ways = dp_prev[r][offset]  # diff==0 means index offset.
        # Multiply by r! * c! to get count of balanced permutations.
        ans = fact[r] * fact[c] % MOD
        ans = ans * valid_ways % MOD
        return ans