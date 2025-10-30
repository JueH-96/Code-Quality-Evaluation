# YOUR CODE HERE
import sys
import threading
import numpy as np
def main():
    import sys
    import numpy as np
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = list(map(int, N_and_rest[1:N+1]))
    max_Ai = max(A_list)
    max_s = max_Ai * 2

    # Precompute f(s) for s up to max_s
    max_s_p2 = 1
    while max_s_p2 <= max_s:
        max_s_p2 <<= 1

    f_s = np.zeros(max_s_p2, dtype=np.int64)

    # Precompute f(s) for s up to max_s
    for s in range(1, max_s + 1):
        x = s
        while x % 2 == 0:
            x //= 2
        f_s[s] = x

    counts_Ai = np.zeros(max_s_p2, dtype=np.float64)

    # Build counts_Ai
    for Ai in A_list:
        counts_Ai[Ai] += 1

    # Perform FFT
    from numpy.fft import fft, ifft
    fft_counts_Ai = fft(counts_Ai)
    fft_counts_s = fft_counts_Ai * fft_counts_Ai
    counts_s = ifft(fft_counts_s).real

    # Round counts_s to nearest integer
    counts_s = np.rint(counts_s).astype(np.int64)

    # Since counts_s[s] includes both (i,j) and (j,i), total pairs where i <= j is counts_s[s] + counts_diag[s]) // 2
    # But we can accept counts_pairs[s] = counts_s[s] // 2

    counts_s = counts_s[:max_s + 1]
    counts_pairs = counts_s // 2

    # Compute total_sum
    total_sum = 0
    for s in range(2, max_s + 1):
        cp = counts_pairs[s]
        if cp > 0:
            total_sum += cp * f_s[s]

    # Print the answer
    print(total_sum)
threading.Thread(target=main).start()