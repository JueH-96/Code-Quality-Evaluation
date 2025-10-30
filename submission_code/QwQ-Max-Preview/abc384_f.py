import numpy as np
from collections import defaultdict

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    freq = defaultdict(int)
    max_a = 0
    for x in A:
        freq[x] += 1
        if x > max_a:
            max_a = x
    max_sum = 2 * max_a
    
    # Compute the next power of two greater than or equal to max_sum
    n = 1
    while n <= max_sum:
        n *= 2
    
    # Create frequency array of size 'n' initialized to zero
    freq_array = np.zeros(n, dtype=np.complex128)
    for x in freq:
        freq_array[x] = freq[x]
    
    # Compute FFT of the frequency array
    fft_freq = np.fft.fft(freq_array)
    
    # Compute the square of the FFT to get the convolution
    convolution = np.fft.ifft(fft_freq ** 2).real.round().astype(int)
    
    # Precompute count_single for each s which is 2*x
    count_single = defaultdict(int)
    for x in freq:
        s = 2 * x
        if s > max_sum:
            continue
        count_single[s] = freq[x]
    
    total = 0
    for s in range(max_sum + 1):
        ordered_pairs = convolution[s]
        current_count = (ordered_pairs + count_single.get(s, 0)) // 2
        if current_count <= 0:
            continue
        # Compute the largest odd divisor of s
        temp = s
        if temp == 0:
            continue
        k = 0
        while temp % 2 == 0:
            k += 1
            temp >>= 1
        f_s = s >> k
        total += current_count * f_s
    
    print(int(total))

if __name__ == '__main__':
    main()