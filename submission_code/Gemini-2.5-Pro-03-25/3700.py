import sys
from collections import Counter
from typing import List

# Optional: Increase recursion depth if needed, although this solution is iterative.
# sys.setrecursionlimit(2000) 

class Solution:
    """
    Finds the number of subsequences of size 5 of nums with a unique middle mode.
    A subsequence of size 5 has a unique middle mode if its middle element (at index 2)
    appears strictly more times than any other element in the subsequence.
    Returns the count modulo 10^9 + 7.
    """
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        
        N = len(nums)
        # Need at least 5 elements to form a subsequence of size 5.
        if N < 5:
            return 0
        MOD = 10**9 + 7

        # Precompute factorials and their modular inverses for combinations nCr % MOD
        # MAX_N_COMB should be N+1 because we need factorials up to N.
        MAX_N_COMB = N + 1
        fact = [1] * MAX_N_COMB
        invfact = [1] * MAX_N_COMB
        for i in range(1, MAX_N_COMB):
            fact[i] = (fact[i-1] * i) % MOD

        # Calculate modular inverse using Fermat's Little Theorem (since MOD is prime)
        # pow(a, m-2, m) computes a^(m-2) mod m, which is the modular inverse of a mod m.
        invfact[MAX_N_COMB-1] = pow(fact[MAX_N_COMB-1], MOD - 2, MOD)
        for i in range(MAX_N_COMB - 2, -1, -1):
            # Compute inverse factorials iteratively using the relation: invfact[i] = invfact[i+1] * (i+1) mod MOD
            invfact[i] = (invfact[i+1] * (i+1)) % MOD

        def nCr_mod(n, r):
            """ Calculates nCr modulo MOD using precomputed factorials and inverses """
            # If r is out of valid range [0, n], combinations is 0.
            if r < 0 or r > n:
                return 0
            
            # Compute nCr = n! / (r! * (n-r)!) mod MOD
            # Using properties: n! * (r!)^(-1) * ((n-r)!)^(-1) mod MOD
            num = fact[n]
            den = (invfact[r] * invfact[n-r]) % MOD
            return (num * den) % MOD
        
        total_subsequences = 0
        
        # Iterate through each index `i` potentially serving as the middle element's index (i_2) in the subsequence
        for i in range(N):
            x = nums[i] # The value of the potential middle element s_2 = x
            
            # Compute frequency map LCounts for elements to the left of index i (nums[0..i-1])
            LCounts = Counter()
            for k in range(i):
                 LCounts[nums[k]] += 1
            
            # Compute frequency map RCounts for elements to the right of index i (nums[i+1..N-1])
            RCounts = Counter()
            for k in range(i + 1, N):
                 RCounts[nums[k]] += 1
            
            # Counts of x on Left and Right sides relative to index i
            L_x = LCounts.get(x, 0)
            R_x = RCounts.get(x, 0)
            
            # Total counts of elements on Left and Right sides
            L_total_count = i 
            R_total_count = N - 1 - i 

            # To form a subsequence of size 5 with middle element at index i,
            # we need to choose 2 elements from the left and 2 elements from the right.
            # If not possible (i.e., < 2 elements on either side), skip this index i.
            if L_total_count < 2 or R_total_count < 2:
                continue 

            # Counts of non-x elements on Left and Right sides
            L_neg_x = L_total_count - L_x
            R_neg_x = R_total_count - R_x

            # Accumulator for subsequences centered at index i with value x
            current_sum_for_i = 0 

            # Case Nx = 5: Frequency of x in subsequence is 5. This means all 5 elements are x.
            # We must choose 2 x's from left, 2 x's from right. The middle element is x.
            C5 = (nCr_mod(L_x, 2) * nCr_mod(R_x, 2)) % MOD
            current_sum_for_i = (current_sum_for_i + C5) % MOD

            # Case Nx = 4: Frequency of x is 4. Subsequence has four x's and one non-x element y.
            # This always satisfies unique middle mode condition (4 > 1).
            # Config 1: Choose 1 x left, 1 non-x left; 2 x right, 0 non-x right
            term1_C4 = (nCr_mod(L_x, 1) * nCr_mod(L_neg_x, 1)) % MOD
            term1_C4 = (term1_C4 * nCr_mod(R_x, 2)) % MOD

            # Config 2: Choose 2 x left, 0 non-x left; 1 x right, 1 non-x right
            term2_C4 = (nCr_mod(L_x, 2) * nCr_mod(R_x, 1)) % MOD
            term2_C4 = (term2_C4 * nCr_mod(R_neg_x, 1)) % MOD
            
            C4 = (term1_C4 + term2_C4) % MOD
            current_sum_for_i = (current_sum_for_i + C4) % MOD

            # Case Nx = 3: Frequency of x is 3. Subsequence has three x's and two non-x elements y, z.
            # If y=z, freq(x)=3, freq(y)=2. Unique mode x. OK.
            # If y!=z, freq(x)=3, freq(y)=1, freq(z)=1. Unique mode x. OK.
            # This case always satisfies the condition.
            # Config 1: Choose 0 x left, 2 non-x left; 2 x right, 0 non-x right
            term1_C3 = (nCr_mod(L_neg_x, 2) * nCr_mod(R_x, 2)) % MOD
             
            # Config 2: Choose 1 x left, 1 non-x left; 1 x right, 1 non-x right
            term2_C3 = (nCr_mod(L_x, 1) * nCr_mod(L_neg_x, 1)) % MOD
            term2_C3 = (term2_C3 * nCr_mod(R_x, 1)) % MOD
            term2_C3 = (term2_C3 * nCr_mod(R_neg_x, 1)) % MOD

            # Config 3: Choose 2 x left, 0 non-x left; 0 x right, 2 non-x right
            term3_C3 = (nCr_mod(L_x, 2) * nCr_mod(R_neg_x, 2)) % MOD

            C3 = (term1_C3 + term2_C3 + term3_C3) % MOD
            current_sum_for_i = (current_sum_for_i + C3) % MOD

            # Case Nx = 2: Frequency of x is 2. Subsequence has two x's and three non-x elements y, z, w.
            # Condition: x must be unique mode. Requires freq(x)=2 > freq(y), freq(z), freq(w).
            # This implies y, z, w must be distinct, and distinct from x. All frequencies must be 1 for non-x elements.
            # Calculate total ways to form Nx=2 subsequences first, then subtract invalid cases.
            
            # Calculate total ways C2_total:
            # Config 1: Pick 1 x left, 1 non-x left; 0 x right, 2 non-x right
            term1_C2_total = (nCr_mod(L_x, 1) * nCr_mod(L_neg_x, 1)) % MOD
            term1_C2_total = (term1_C2_total * nCr_mod(R_neg_x, 2)) % MOD
            
            # Config 2: Pick 0 x left, 2 non-x left; 1 x right, 1 non-x right
            term2_C2_total = (nCr_mod(L_neg_x, 2) * nCr_mod(R_x, 1)) % MOD
            term2_C2_total = (term2_C2_total * nCr_mod(R_neg_x, 1)) % MOD
            
            C2_total = (term1_C2_total + term2_C2_total) % MOD

            # Calculate corrections A and B for invalid cases where x is not unique mode
            A = 0 # Count for Type A invalid cases: three identical non-x elements (y, y, y). Freq(x)=2, Freq(y)=3. Mode is y.
            B = 0 # Count for Type B invalid cases: two identical non-x elements and one different (y, y, z). Freq(x)=2, Freq(y)=2, Freq(z)=1. Modes x and y.

            # Iterate over all distinct values `y` present either left or right of `i` to compute corrections.
            # Using set union of keys from LCounts and RCounts ensures we cover all relevant `y`.
            for y in LCounts.keys() | RCounts.keys(): 
                 if y == x: continue # Correction involves non-x elements only.
                 
                 L_y = LCounts.get(y, 0)
                 R_y = RCounts.get(y, 0)

                 # Type A correction calculation for value y: Find ways to pick three y's.
                 # Config A1: Pick 1 x left, requires 1 y left and 2 y's right.
                 term1_A_y = (nCr_mod(L_x, 1) * nCr_mod(L_y, 1)) % MOD 
                 term1_A_y = (term1_A_y * nCr_mod(R_y, 2)) % MOD 
                 
                 # Config A2: Pick 1 x right, requires 2 y's left and 1 y right.
                 term2_A_y = (nCr_mod(R_x, 1) * nCr_mod(L_y, 2)) % MOD 
                 term2_A_y = (term2_A_y * nCr_mod(R_y, 1)) % MOD 
                 
                 A = (A + term1_A_y + term2_A_y) % MOD

                 # Type B correction calculation involving y: Find ways to pick two y's and one z (z != x, z != y).
                 # The formula derived sums contributions over all possible z != x, y implicitly.
                 
                 # K1(y): Component derived from B_yz expansion, relates to picking z from Left.
                 K1_y_term1 = (nCr_mod(R_x, 1) * nCr_mod(L_y, 2)) % MOD 
                 K1_y_term2 = (nCr_mod(R_x, 1) * nCr_mod(L_y, 1)) % MOD 
                 K1_y_term2 = (K1_y_term2 * nCr_mod(R_y, 1)) % MOD     
                 K1_y_term3 = (nCr_mod(L_x, 1) * nCr_mod(R_y, 2)) % MOD 
                 K1_y = (K1_y_term1 + K1_y_term2 + K1_y_term3) % MOD
                 
                 # K2(y): Component derived from B_yz expansion, relates to picking z from Right.
                 K2_y = (nCr_mod(L_x, 1) * nCr_mod(L_y, 1)) % MOD 
                 K2_y = (K2_y * nCr_mod(R_y, 1)) % MOD     
                 
                 # Sum L_z over z != x, y is L_neg_x - L_y
                 L_neg_x_neg_y = (L_neg_x - L_y + MOD) % MOD # Use +MOD to handle potential negative intermediate result
                 # Sum R_z over z != x, y is R_neg_x - R_y
                 R_neg_x_neg_y = (R_neg_x - R_y + MOD) % MOD 

                 # Total contribution to B for pairs involving y (where y appears twice)
                 term_B_y = (K1_y * L_neg_x_neg_y) % MOD 
                 term_B_y = (term_B_y + K2_y * R_neg_x_neg_y) % MOD 
                 
                 B = (B + term_B_y) % MOD

            # Calculate final valid count for Nx = 2 case by subtracting invalid counts A and B.
            # Add 2*MOD before final modulo to handle potential negative result from subtractions.
            C2 = (C2_total - A - B + 2 * MOD) % MOD 
            current_sum_for_i = (current_sum_for_i + C2) % MOD
            
            # Add the total valid count for subsequences centered at index i with value x
            total_subsequences = (total_subsequences + current_sum_for_i) % MOD

        return total_subsequences