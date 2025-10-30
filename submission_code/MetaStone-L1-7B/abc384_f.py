import numpy as np

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    
    max_s = 2 * 10**7
    freq = np.zeros(max_s + 1, dtype=int)
    for x in A:
        freq[x] += 1
    
    # Compute f(s) for all s up to max_s
    f = np.zeros(max_s + 1, dtype=int)
    for s in range(1, max_s + 1):
        while s % 2 == 0:
            s = s // 2
        f[s] = s
    
    # Compute FFT of f and freq
    fft_f = np.fft.fft(f)
    fft_freq = np.fft.fft(freq)
    
    # Compute the circular convolution
    fft_conv = fft_f * fft_freq
    conv = np.fft.ifft(fft_conv).real
    
    # Round the convolution results to the nearest integer
    conv = np.round(conv).astype(int)
    
    # Compute S_total
    S_total = 0
    for x in A:
        S_total += conv[x]
    
    # Compute D
    D = sum(f[x] for x in A)
    
    # Compute S
    S = (S_total + D) // 2
    
    print(S)

if __name__ == '__main__':
    main()