import sys
import cmath

# Function to perform Cooley-Tukey FFT
# a: list of complex numbers representing polynomial coefficients
# invert: boolean, True for inverse FFT, False for forward FFT
def fft(a, invert):
    n = len(a)
    if n == 1:
        return

    # Divide step: split into even and odd indexed elements
    a0 = a[0::2]
    a1 = a[1::2]

    # Conquer step: recursive calls on sub-problems
    fft(a0, invert)
    fft(a1, invert)

    # Combine step: merge results from sub-problems
    # ang defines the argument for the complex roots of unity
    ang = 2 * cmath.pi / n * (-1 if invert else 1)
    w = 1  # current root of unity
    wn = cmath.exp(1j * ang)  # primitive n-th root of unity

    for i in range(n // 2):
        # Butterfly operation
        a[i] = a0[i] + w * a1[i]
        a[i + n // 2] = a0[i] - w * a1[i]
        w *= wn
    
    # Scale if performing inverse FFT
    if invert:
        for i in range(n):
            a[i] /= 2

# Function to multiply two polynomials using FFT
# a_coeffs: list of coefficients for polynomial A
# b_coeffs: list of coefficients for polynomial B
# Returns list of coefficients for A * B
def multiply_polynomials(a_coeffs, b_coeffs):
    # Determine the smallest power of 2 for FFT size
    # The degree of the product polynomial is (deg(A) + deg(B)).
    # So, we need space for (len(a_coeffs) + len(b_coeffs) - 1) coefficients.
    n = 1
    target_len = len(a_coeffs) + len(b_coeffs) - 1
    while n < target_len:
        n <<= 1  # n = n * 2

    # Pad coefficients with zeros to match the FFT size
    a_padded = list(a_coeffs) + [0] * (n - len(a_coeffs))
    b_padded = list(b_coeffs) + [0] * (n - len(b_coeffs))

    # Apply forward FFT to transform polynomials into frequency domain
    fft(a_padded, False)
    fft(b_padded, False)

    # Multiply corresponding terms in the frequency domain (point-wise multiplication)
    c_transformed = [a_padded[i] * b_padded[i] for i in range(n)]

    # Apply inverse FFT to transform result back to time domain (coefficient form)
    fft(c_transformed, True)

    # Convert complex results back to real integers and round them
    # Rounding is necessary due to floating-point arithmetic in FFT
    res = [round(x.real) for x in c_transformed]
    
    # The result might have trailing zeros beyond the actual degree, trim them
    return res[:target_len]

def solve():
    N = int(sys.stdin.readline())
    # Read the S_i values into a list
    S_values = list(map(int, sys.stdin.readline().split()))

    # If N is less than 3, no triplet can be formed
    if N < 3:
        print(0)
        return

    # Determine the maximum value in S. This helps size arrays efficiently.
    # Max possible S_i is 10^6.
    max_s_val = 0
    if S_values: # Check if S_values is not empty, though N >= 3 implies it won't be
        max_s_val = max(S_values)
    
    # Create the coefficient list for polynomial P(x) = sum_{s_i in S} x^(s_i)
    # coeffs_P[k] will be 1 if k is in S, else 0.
    # Array size is max_s_val + 1 to accommodate index up to max_s_val.
    coeffs_P = [0] * (max_s_val + 1)
    for s_val in S_values:
        coeffs_P[s_val] = 1

    # Compute P(x)^2 using the polynomial multiplication function (which uses FFT)
    # coeffs_Q[k] will store the count of ordered pairs (i, j) where i, j are in S and i + j = k.
    coeffs_Q = multiply_polynomials(coeffs_P, coeffs_P)

    total_fine_triplets = 0

    # Iterate through each value B_val present in S
    # B_val will be the potential middle element of a fine triplet (A, B_val, C)
    for B_val in S_values:
        # A fine triplet satisfies A < B_val < C and A + C = 2 * B_val.
        target_sum = 2 * B_val
        
        # Check if target_sum is a valid index for coeffs_Q.
        # It should always be if B_val <= max_s_val, as max sum is 2*max_s_val
        if target_sum < len(coeffs_Q):
            # num_pairs_for_sum is the count of ordered pairs (A, C) from S that sum to target_sum.
            num_pairs_for_sum = coeffs_Q[target_sum]

            # The pair (B_val, B_val) sums to 2 * B_val. This specific pair contributes 1 to num_pairs_for_sum.
            # However, for a fine triplet (A, B, C), A, B, C must be distinct. So A != B_val and C != B_val.
            # Therefore, the (B_val, B_val) pair is not a valid (A, C) for our purpose and must be excluded.
            # We subtract 1 to remove its contribution.
            num_pairs_for_sum -= 1
            
            # After subtracting 1, num_pairs_for_sum counts ordered pairs (A, C) where A + C = 2 * B_val and A != C.
            # Since A != C, for every distinct unordered pair {A, C} (where A < C), both (A, C) and (C, A) are
            # counted in num_pairs_for_sum.
            # So, to get the count of unique unordered pairs {A, C} that can form a triplet, we divide by 2.
            # These unique pairs, combined with B_val, form the desired fine triplets (A, B_val, C).
            total_fine_triplets += num_pairs_for_sum // 2

    print(total_fine_triplets)

if __name__ == '__main__':
    solve()