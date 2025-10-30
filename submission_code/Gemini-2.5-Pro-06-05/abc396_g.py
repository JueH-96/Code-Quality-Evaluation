# YOUR CODE HERE
import sys

def main():
    """
    Main function to read input, solve the problem using FWHT, and print the output.
    """
    try:
        H, W = map(int, sys.stdin.readline().split())
    except ValueError:
        # Handles potential empty lines at the end of input.
        return

    # N represents the number of possible row patterns, which is 2^W.
    N = 1 << W
    
    # counts[i] stores the number of times a row with integer representation 'i'
    # appears in the input grid.
    counts = [0] * N
    for _ in range(H):
        row_str = sys.stdin.readline().strip()
        if not row_str: continue
        row_int = int(row_str, 2)
        counts[row_int] += 1
        
    # g[i] stores the minimum number of 1s a row pattern 'i' can have.
    # This is achieved by either keeping it as is or flipping it.
    # A flip changes the number of 1s from popcount(i) to W - popcount(i).
    g = [0] * N
    for i in range(N):
        popcount = bin(i).count('1')
        g[i] = min(popcount, W - popcount)

    def fwht_inplace(a):
        """
        Performs the Fast Walsh-Hadamard Transform in-place.
        This version is for XOR convolution.
        """
        n = len(a)
        h = 1
        while h < n:
            for i in range(0, n, h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2

    # The problem is to find min_C sum_R counts[R] * g(R xor C).
    # This is the XOR convolution of `counts` and `g`.
    # It can be computed efficiently using FWHT:
    # conv(counts, g) = IFWHT(FWHT(counts) * FWHT(g))
    # where * is element-wise product.
    # The property FWHT(FWHT(x)) = N * x lets us use FWHT for the inverse transform.

    # 1. Transform both `counts` and `g` arrays.
    fwht_inplace(counts)
    fwht_inplace(g)
    
    # 2. Compute the element-wise product of the transformed arrays.
    # This corresponds to the FWHT of the convolution result.
    s_conv_transformed = [counts[i] * g[i] for i in range(N)]
    
    # 3. Apply FWHT again. This gives N * (actual convolution result).
    fwht_inplace(s_conv_transformed)
    
    # The i-th element of the convolution result is the total number of ones
    # in the grid if we apply the column flip pattern 'i'.
    # We want the minimum of these values.
    # All resulting values must be non-negative.
    min_sum = min(s_conv_transformed) // N

    print(min_sum)

if __name__ == "__main__":
    main()