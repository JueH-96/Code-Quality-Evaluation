import sys
import numpy as np

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    MAX_A_VAL = 10**7
    MAX_SUM_VAL = 2 * MAX_A_VAL

    # Step 1: Precompute f(x) = x / 2^(v_2(x)) for x up to MAX_SUM_VAL
    # f_values[x] stores f(x)
    f_values = [0] * (MAX_SUM_VAL + 1)
    for i in range(1, MAX_SUM_VAL + 1):
        # f(x) is x divided by the largest power of 2 that divides x
        # This is equivalent to x // (x & -x)
        f_values[i] = i // (i & -i)

    # Step 2: Count occurrences of each number in A
    # `counts_by_val[x]` stores the count of `x` in A
    # Using numpy array for efficiency and compatibility with numpy.fft
    counts_by_val = np.zeros(MAX_A_VAL + 1, dtype=np.int64)
    for x in A:
        counts_by_val[x] += 1

    total_sum = 0

    # Step 3: Calculate sum_{i=1}^N f(A_i+A_i) = sum_{i=1}^N f(A_i)
    for x in A:
        total_sum += f_values[x]

    # Step 4: Calculate sum_{i=1}^N sum_{j=i+1}^N f(A_i+A_j) using FFT
    
    # Determine the smallest power of 2 greater than MAX_SUM_VAL for FFT size
    fft_size = 1
    while fft_size <= MAX_SUM_VAL:
        fft_size <<= 1
    
    # Convert counts_by_val to float for FFT computation
    counts_by_val_float = counts_by_val.astype(float)
    
    # Perform FFT: P_fft is the FFT of the counts_by_val array
    P_fft = np.fft.rfft(counts_by_val_float, n=fft_size)
    
    # Multiply FFTs to get FFT of the convolution
    conv_fft = P_fft * P_fft
    
    # Inverse FFT to get the convolution result
    conv_result = np.fft.irfft(conv_fft)
    
    # Round to nearest integer and cast to int64, as counts should be integers
    conv_result = np.round(conv_result).astype(np.int64)

    # Iterate through all possible sums X (from 2 to MAX_SUM_VAL)
    # conv_result[X] stores sum_{a+b=X} counts_by_val[a] * counts_by_val[b]
    # This sum includes ordered pairs (a,b). We need to count unique pairs (u,v) with u<v.
    for X in range(2, MAX_SUM_VAL + 1):
        num_pairs_i_lt_j_sum_to_X = 0
        
        # Calculate term for pairs (a,a) where a+a = X (i.e., X is even)
        # These correspond to A_u = A_v = val (where val = X/2).
        # Number of ways to choose two distinct indices from 'counts_by_val[val]' elements is val_count * (val_count - 1) // 2
        val_at_half_X_count = 0
        if X % 2 == 0:
            val_half_X = X // 2
            if val_half_X <= MAX_A_VAL: # Check if val_half_X is within valid index range
                val_at_half_X_count = counts_by_val[val_half_X]
        
        num_pairs_i_lt_j_sum_to_X += val_at_half_X_count * (val_at_half_X_count - 1) // 2

        # Calculate term for pairs (a,b) where a < b and a+b = X
        # In conv_result[X], these pairs contribute 2 * counts_by_val[a] * counts_by_val[b].
        # The sum of all a=b terms (counts_by_val[a]*counts_by_val[a]) is exactly val_at_half_X_count * val_at_half_X_count.
        # So, (conv_result[X] - (val_at_half_X_count * val_at_half_X_count)) gives the sum of 2*counts_by_val[a]*counts_by_val[b] for a != b.
        # Dividing by 2 gives the sum for a < b.
        
        sum_counts_for_unequal_halves = (conv_result[X] - (val_at_half_X_count * val_at_half_X_count)) // 2
        num_pairs_i_lt_j_sum_to_X += sum_counts_for_unequal_halves

        # Add contribution of these pairs to total_sum
        total_sum += f_values[X] * num_pairs_i_lt_j_sum_to_X

    sys.stdout.write(str(total_sum) + "
")

solve()