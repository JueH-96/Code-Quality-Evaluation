# YOUR CODE HERE
import sys
import numpy as np
import math # Not strictly needed, but good practice if math functions were used

# Function to compute v2(x) and f(x)
# v2(x) is the exponent k such that x = 2^k * m, where m is odd. This is the exponent of the highest power of 2 dividing x.
# f(x) = m is the odd part of x.
def get_v2_f(x):
    """
    Computes the exponent of the highest power of 2 dividing x (v2(x))
    and the odd part of x (f(x)).
    Requires x > 0.
    """
    # Find k = v2(x) using bit manipulation: x & -x isolates the lowest set bit.
    # The value is 2^k. Its bit_length() is k+1. So k = bit_length() - 1.
    # Example: x=24 (11000_2). -x is (~x + 1). In 2's complement, -24 is (...11101000).
    # x & -x = (0...011000) & (...11101000) = 0...001000 = 8 = 2^3.
    # (8).bit_length() = 4. So k = 4 - 1 = 3.
    # This correctly identifies that the lowest set bit is at position 3 (0-indexed).
    k = (x & -x).bit_length() - 1 
    # Right shift x by k bits to get the odd part m. m = x / 2^k.
    m = x >> k
    return k, m

def solve():
    N = int(sys.stdin.readline())
    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))

    vals = []
    MAX_A_val = 0
    # Precompute k=v2(A_i) and m=f(A_i) for all A_i
    for x in A:
        # Constraints state A_i >= 1, so x > 0.
        k, m = get_v2_f(x)
        vals.append({'k': k, 'm': m})
        MAX_A_val = max(MAX_A_val, x) # Track max A_i value

    # Group indices based on the value of k=v2(A_i)
    groups = {}
    MAX_M_val = 0 # Track max odd part M_i encountered
    for i in range(N):
        k = vals[i]['k']
        m = vals[i]['m']
        if k not in groups:
            groups[k] = []
        groups[k].append(m)
        MAX_M_val = max(MAX_M_val, m) 

    # Compute counts and sum of M values for each group
    Count = {}
    SumM = {}
    # Get sorted list of k values present. Sorting helps structured processing (e.g., Part2Sum).
    k_list = sorted(groups.keys()) 

    for k in k_list:
        Count[k] = len(groups[k])
        # Use Python's arbitrary precision integers for potentially large sums
        SumM[k] = sum(groups[k]) 

    # Compute TotalM = Sum_{i=1..N} f(A_i) = Sum_{i=1..N} M_i
    TotalM = sum(SumM.values()) 

    # Calculate Part2Sum: contribution from pairs (i, j) where v2(A_i) != v2(A_j)
    # This sums f(A_i+A_j) for pairs i in S_k1, j in S_k2 where k1 < k2.
    # f(A_i+A_j) = M_i + 2^(k2-k1) * M_j if k1 < k2
    # f(A_i+A_j) = M_j + 2^(k1-k2) * M_i if k2 < k1
    # The final sum S requires sum over i <= j.
    # We compute DoubleS = Sum_{i,j} f(A_i+A_j) and use S = (DoubleS + TotalM) / 2.
    # DoubleS = Sum_{k} DoubleSum_k + 2 * Part2Sum, where Part2Sum calculation is below.
    Part2Sum = 0
    for idx1 in range(len(k_list)):
        for idx2 in range(idx1 + 1, len(k_list)):
            k1 = k_list[idx1]
            k2 = k_list[idx2]
            
            # Total contribution from pairs (i, j) where i in S_k1, j in S_k2
            term1 = Count[k2] * SumM[k1] # Sum of M_i part over all pairs
            term2 = Count[k1] * (1 << (k2 - k1)) * SumM[k2] # Sum of 2^(k2-k1) * M_j part over all pairs
            Part2Sum += term1 + term2
            # The sum for k1 > k2 gives the same expression by symmetry.

    # Calculate total contribution from pairs within the same group k: Sum_k DoubleSum_k
    # DoubleSum_k = Sum_{i, j in S_k} f(A_i+A_j) = Sum_{i, j in S_k} f(M_i+M_j)
    Total_IntraGroup_DoubleSum = 0

    # Determine K_max based on max possible M_i + M_j sum
    # This determines the maximum exponent 'e' we need consider in f(M_i+M_j) = (M_i+M_j)/2^e
    max_possible_M_sum = 2 * MAX_M_val
    if max_possible_M_sum == 0:
         K_max_exponent = 0 
    else:
         # Max possible exponent e = v2(M_i+M_j). Max value is floor(log2(max_M_sum)).
         # The bit_length gives 1 + position of highest set bit. max_exponent is this position.
         K_max_exponent = max_possible_M_sum.bit_length() -1
         if K_max_exponent < 0: K_max_exponent = 0 # Handle edge case if max_M_val=0
    
    # FFT length must be a power of 2 >= max_possible_M_sum + 1 to avoid circular convolution effects.
    FFT_LEN_min = max_possible_M_sum + 1
    FFT_LEN = 1
    while FFT_LEN < FFT_LEN_min:
        FFT_LEN <<= 1
    
    # The K_max for S_e summation loop is the max possible exponent e=v2(M_i+M_j).
    K_max_for_sum = K_max_exponent 

    # Pre-allocate NumPy array for efficiency. Use dtype=object to handle potentially large integers.
    zeros_fft_len = np.zeros(FFT_LEN, dtype=object)

    # Iterate through each group k to calculate its DoubleSum_k contribution
    for k in k_list:
        N_k = Count[k]
        if N_k == 0: continue # Skip empty groups

        M_list = groups[k]
        
        # Initialize polynomial coefficient arrays using pre-allocated zeros array
        P0_coeffs = zeros_fft_len.copy()
        P1_coeffs = zeros_fft_len.copy()
        
        # Aggregate counts of each distinct M value in the group for efficiency
        term_counts = {}
        for m_val in M_list:
             term_counts[m_val] = term_counts.get(m_val, 0) + 1
        
        # Build polynomial coefficients P0(x) = sum c_p x^{m_p} and P1(x) = sum c_p m_p x^{m_p}
        # c_p is the count of value m_p in the group.
        for m_val, count in term_counts.items():
             # Index corresponds to the M value (exponent of x)
             P0_coeffs[m_val] += count
             P1_coeffs[m_val] += count * m_val

        # Compute convolution 2 * P0 * P1 using FFT
        # The result represents polynomial whose coefficients Y are Sum_{p,q: mp+mq=Y mod FFT_LEN} c_p c_q (mp+mq)
        fft_P0 = np.fft.fft(P0_coeffs)
        fft_P1 = np.fft.fft(P1_coeffs)
        
        # Product in frequency domain corresponds to convolution in time domain
        fft_prod = fft_P0 * fft_P1 * 2
        
        # Inverse FFT to get the coefficients of the resulting polynomial
        result_coeffs = np.fft.ifft(fft_prod)
        
        # S_e[e] stores Sum_{pairs p,q: v2(m_p+m_q)=e} c_p c_q (m_p+m_q)
        # Initialize S_e array for this group k
        S_e = np.zeros(K_max_for_sum + 1, dtype=object)
        
        # Extract coefficients and sum them up based on v2(Y)
        # We only need to check indices Y up to the maximum possible sum M_i+M_j
        for Y in range(1, max_possible_M_sum + 1): 
            coeff_val_complex = result_coeffs[Y]
            # Filter out negligible values that might arise from floating point inaccuracies
            # Check magnitude of both real and imaginary parts
            if abs(coeff_val_complex.real) > 1e-9 or abs(coeff_val_complex.imag) > 1e-9:
                 # Round the real part to get the integer coefficient. Imaginary part should be near zero for real convolution.
                 coeff_val = np.round(coeff_val_complex.real).astype(object)
                 if coeff_val == 0: continue # Skip if it rounds to zero

                 # Find e = v2(Y), the exponent of 2 dividing the sum Y
                 # Y must be positive for this v2 calculation to be well-defined
                 e = (Y & -Y).bit_length() - 1
                 # Check if e is within the valid range [1, K_max_for_sum]
                 # M_i, M_j are odd, so M_i+M_j is even, thus e >= 1.
                 if e >= 1 and e <= K_max_for_sum:
                         # The coefficient Coeff_Y approx Sum_{p,q: mp+mq=Y} c_p c_q (mp+mq).
                         # S_e = sum_{Y: v2(Y)=e} Coeff_Y
                         S_e[e] += coeff_val

        # Compute DoubleSum_k = sum_{e=1..K_max} S_e / 2^e
        current_DoubleSum_k = 0
        for e in range(1, K_max_for_sum + 1):
             if S_e[e] != 0:
                  # Use integer division //
                  quotient = S_e[e] // (1 << e)
                  current_DoubleSum_k += quotient
        
        # Add this group's contribution to the total sum over pairs within the same group
        Total_IntraGroup_DoubleSum += current_DoubleSum_k

    # Final Answer computation using the formula S = 1/2 * (DoubleS + TotalM)
    # DoubleS = Sum_{i,j} f(A_i+A_j) = (Sum over intra-group pairs) + (Sum over inter-group pairs)
    # DoubleS = Total_IntraGroup_DoubleSum + 2 * Part2Sum (factor 2 accounts for i<j and i>j pairs)
    DoubleS = Total_IntraGroup_DoubleSum + 2 * Part2Sum
    
    # The desired sum S = Sum_{i<=j} f(A_i+A_j)
    FinalAnswer = (DoubleS + TotalM) // 2 # Use integer division //
    
    print(FinalAnswer)

# Execute the solve function to read input, compute, and print the answer.
solve()