# YOUR CODE HERE
import sys
import numpy as np

def solve():
    """
    Reads input, computes the number of fine triplets using FFT, and prints the result.
    A fine triplet (A, B, C) satisfies A, B, C are in the input set S, A < B < C, and B - A = C - B (or A + C = 2B).

    The approach uses Fast Fourier Transform (FFT) to efficiently compute the convolution 
    of the characteristic polynomial of the set S with itself.
    Let S be the input set of distinct positive integers.
    Define a polynomial P(x) = sum_{i in S} x^i.
    The square of this polynomial is P(x)^2 = (sum_{a in S} x^a) * (sum_{c in S} x^c) = sum_{k} C(k) x^k.
    The coefficient C(k) represents the number of pairs (a, c) such that a is in S, c is in S, and a + c = k.

    We are looking for triplets (A, B, C) such that A, B, C are in S, A < B < C, and A + C = 2B.
    Consider the sum S_fft = sum_{B in S} C(2B).
    C(2B) is the number of pairs (A, C) in S such that A + C = 2B.
    This sum relates to the number of fine triplets (N_triplets) as follows:
    Each fine triplet (A, B, C) where A < B < C contributes 2 to the sum S_fft. This is because the pairs (A, C) and (C, A) are counted in C(2B) when we consider the middle element B.
    Additionally, for each element B in S, the pair (B, B) contributes 1 to C(2B). So, each B in S contributes 1 to the sum S_fft.
    Therefore, S_fft = 2 * N_triplets + N, where N is the number of elements in S.
    We can find N_triplets = (S_fft - N) // 2.
    """
    
    # Read N, the number of elements in the set S
    N = int(sys.stdin.readline())
    # Read the elements of the set S
    # S contains N distinct positive integers.
    S_elements = list(map(int, sys.stdin.readline().split()))

    # Define the maximum possible value for an element in S based on problem constraints
    # S_i <= 1,000,000
    MAX_VAL_CONSTRAINT = 1000000 
    
    # Create the polynomial coefficient array p.
    # p[i] = 1 if integer i is present in the set S, otherwise p[i] = 0.
    # The polynomial is P(x) = sum_{i=0}^{MAX_VAL_CONSTRAINT} p[i] * x^i.
    # The size needs to be MAX_VAL_CONSTRAINT + 1 to accommodate index MAX_VAL_CONSTRAINT.
    p = np.zeros(MAX_VAL_CONSTRAINT + 1, dtype=int) 
    
    # Populate the array p based on the input set S
    # Mark the presence of each element from S in the coefficient array.
    for x in S_elements:
        # Input guarantees 1 <= x <= MAX_VAL_CONSTRAINT
        if 1 <= x <= MAX_VAL_CONSTRAINT:
             p[x] = 1

    # Determine the appropriate length for the FFT computation.
    # The result of the convolution P(x)^2 will have coefficients for powers up to 2 * max(S).
    # The maximum possible value is 2 * MAX_VAL_CONSTRAINT.
    # To use FFT for linear convolution (not circular), the transform length `fft_len` must be at least 
    # degree(P) + degree(P) + 1 = 2 * degree(P) + 1.
    # Here, max degree is MAX_VAL_CONSTRAINT, so we need fft_len >= 2 * MAX_VAL_CONSTRAINT + 1.
    # FFT algorithms are most efficient with lengths that are powers of 2.
    fft_len = 1
    while fft_len <= 2 * MAX_VAL_CONSTRAINT:
        fft_len <<= 1 # Calculate the smallest power of 2 that is > 2 * MAX_VAL_CONSTRAINT

    # Compute the square of the polynomial P(x) using FFT-based convolution.
    
    # 1. Compute the Fast Fourier Transform (FFT) of the coefficient array p.
    #    numpy.fft.fft automatically handles padding the input array `p` to `fft_len` if `fft_len` is provided and larger than `len(p)`.
    P_fft = np.fft.fft(p, fft_len)
    
    # 2. Square the result element-wise in the frequency domain.
    #    By the convolution theorem, multiplication in the frequency domain corresponds to convolution in the time domain.
    P2_fft = P_fft * P_fft
    
    # 3. Compute the Inverse Fast Fourier Transform (IFFT) to transform back to the coefficient domain.
    p_squared_padded = np.fft.ifft(P2_fft)

    # Extract the coefficients of P(x)^2. The result of IFFT should ideally be real integers representing counts.
    # Due to floating-point inaccuracies inherent in FFT algorithms, results might be complex numbers with small imaginary parts.
    # We take the real part, round it to the nearest integer to correct for potential floating point errors, 
    # and convert to np.int64 to safely handle potentially large counts without overflow.
    p_squared = np.round(p_squared_padded.real).astype(np.int64)

    # Calculate the sum S_fft = sum_{B in S} C(2B), where C(k) is the coefficient of x^k in P(x)^2.
    # C(2B) = p_squared[2*B] gives the number of pairs (A, C) from S such that A + C = 2B.
    S_fft = 0
    
    # Iterate through each element B that is present in the input set S.
    for B in S_elements: 
        # Calculate the index corresponding to the term x^(2B) in P(x)^2.
        idx = 2 * B
        
        # Access the pre-computed coefficient C(2B) from the p_squared array.
        # Check index boundary: idx must be less than fft_len. 
        # This is guaranteed because B <= MAX_VAL_CONSTRAINT implies 2*B <= 2*MAX_VAL_CONSTRAINT, 
        # and we chose fft_len > 2*MAX_VAL_CONSTRAINT.
        if idx < len(p_squared): # Safety check, though technically redundant if fft_len is correct.
            S_fft += p_squared[idx]

    # Relate the computed sum S_fft to the number of fine triplets (N_triplets).
    # S_fft = 2 * N_triplets + N, where N is the number of elements in the input set S.
    # N_triplets counts triplets (A, B, C) with A, B, C in S and A < B < C.
    # The term 'N' accounts for the contribution of pairs (B, B) for each B in S.
    # Solve for N_triplets using integer division (//).
    N_triplets = (S_fft - N) // 2

    # Print the final count of fine triplets to standard output.
    print(N_triplets)

# Execute the main logic contained within the solve function when the script is run.
solve()