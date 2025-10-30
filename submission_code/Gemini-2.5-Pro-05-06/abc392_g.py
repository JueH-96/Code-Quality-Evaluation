import sys
import numpy as np

def solve():
    N = int(sys.stdin.readline())
    # Reading S_values could be slow if N is large.
    # Optimized reading:
    s_values_str = sys.stdin.readline().split()
    S_values = [int(x) for x in s_values_str]

    if N < 3:
        print(0)
        return

    V_max = 0
    # max() on a list can be slow if it's not already a Python list.
    # Iterating once to find max and populate poly_coeffs can be combined,
    # but let's find V_max first for clarity and array sizing.
    for x in S_values:
        if x > V_max:
            V_max = x
    
    # S_i are positive integers, so V_max >= 1.
    # poly_coeffs[i] = 1.0 if i is in S, 0.0 otherwise.
    # Size is V_max + 1 for indices 0 to V_max.
    poly_coeffs = np.zeros(V_max + 1, dtype=float)
    for val in S_values:
        poly_coeffs[val] = 1.0
    
    # FFT length: smallest power of 2 >= 2*V_max + 1.
    # (K-1).bit_length() gives ceil(log2(K)). So 1 << ((K-1).bit_length()) is smallest power of 2 >= K.
    # Let K_target = 2 * V_max + 1.
    # fft_len = 1 << ( (2 * V_max + 1) - 1 ).bit_length() = 1 << (2 * V_max).bit_length()
    # This handles V_max=0 too (resulting in fft_len=1), though V_max >= 1 here.
    fft_len = 1 << (2 * V_max).bit_length()

    # Compute P(x)^2 using FFT
    fft_P = np.fft.rfft(poly_coeffs, n=fft_len)
    fft_Q = fft_P * fft_P  # Element-wise product in frequency domain
    
    # Inverse FFT to get coefficients of Q(x) = P(x)^2
    coeffs_Q_float = np.fft.irfft(fft_Q, n=fft_len)

    total_fine_triplets = 0
    for B_val in S_values:
        target_sum_2B = 2 * B_val
        
        # The index target_sum_2B must be < len(coeffs_Q_float) (which is fft_len).
        # Since fft_len is chosen >= 2*V_max+1, and max(target_sum_2B) is 2*V_max,
        # this index will always be valid.
        
        # c_2B should be an integer. Round to handle potential float inaccuracies.
        c_2B = round(coeffs_Q_float[target_sum_2B])
        
        # (c_2B - 1) counts 2 times the number of {A,C} pairs with A!=C, A,C in S, A+C=2B.
        # This is because c_2B counts (A,C) and (C,A), plus (B,B) once.
        # Each such {A,C} pair (A<C) gives one fine triplet (A, B_val, C).
        if c_2B > 0: # Ensure c_2B is at least 1 (for the (B_val,B_val) pair)
             total_fine_triplets += (c_2B - 1) // 2
    
    print(total_fine_triplets)

if __name__ == '__main__':
    solve()