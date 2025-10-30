# YOUR CODE HERE
import sys
import numpy as np

# Read N from the first line of stdin
try:
    N = int(sys.stdin.readline())
except (ValueError, IndexError):
    # Handle cases of empty or malformed input
    N = 0

# If N > 0, read the set S from the second line of stdin
if N > 0:
    try:
        S = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        S = []
        N = 0
else:
    S = []

# A fine triplet requires 3 distinct numbers. If we have fewer, the answer is 0.
if N < 3:
    print(0)
else:
    # The problem is to find triplets (A, B, C) from S where A < B < C and A + C = 2B.
    # This is equivalent to finding 3-term arithmetic progressions in S.
    # We use a Fast Fourier Transform (FFT) based approach for efficiency.

    # We represent S as a polynomial P(x) = sum_{s in S} x^s.
    # P(x)^2 will have coefficients q_k that count pairs (a, c) from S summing to k.
    
    # The maximum value in S determines the required FFT size.
    max_val = max(S)
    
    # FFT size must be a power of 2 > degree of P(x)^2, which is 2 * max_val.
    fft_size = 1
    while fft_size <= 2 * max_val:
        fft_size <<= 1

    # Create the coefficient array for P(x).
    P = np.zeros(fft_size)
    for x in S:
        P[x] = 1

    # Compute P(x)^2 using FFT.
    # The coefficients of the resulting polynomial Q(x) are obtained by
    # transforming P(x) to the frequency domain, squaring, and transforming back.
    P_fft = np.fft.fft(P)
    Q_fft = P_fft * P_fft
    Q_coeffs = np.round(np.fft.ifft(Q_fft).real).astype(np.int64)
    
    # For each b in S, Q_coeffs[2*b] gives the number of pairs (a, c) in S
    # such that a + c = 2b.
    # Summing these counts for all b in S gives the total number of (a, b, c) triplets.
    total_triplets = 0
    for b in S:
        total_triplets += Q_coeffs[2 * b]
        
    # This sum counts:
    # 1. Fine triplets (A<B<C) twice (due to pairs (A,C) and (C,A)).
    # 2. Trivial triplets (s,s,s) once (due to the pair (s,s)).
    # So, total_triplets = 2 * (number of fine triplets) + N.
    fine_triplets_count = (total_triplets - N) // 2
    
    print(fine_triplets_count)