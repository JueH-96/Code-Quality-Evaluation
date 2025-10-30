def main():
    import sys, math
    import numpy as np
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # First integer is n.
    n = int(data[0])
    # Read the set S elements, note they are distinct.
    S = [int(x) for x in data[1:]]
    
    # If there are fewer than 3 numbers, no triplet can be formed.
    if n < 3:
        sys.stdout.write("0")
        return

    # Let F be an indicator function: F[x] = 1 if x in S, else 0.
    # S values are up to 10^6. We need to compute the convolution F*F 
    # so that (F*F)[X] = sum_{a+c = X} 1.
    # But a fine triplet (A, B, C) satisfies A < B < C and (B-A)==(C-B),
    # which is equivalent to: B = (A + C) / 2, (A+C is even).
    # For a given B in S, let X = 2*B. Then the number of pairs (A, C) in S 
    # with A+C = X is (F*F)[2B]. But note that each valid unordered pair (A, C) 
    # with A < C appears twice in the convolution sum. Hence the number of fine 
    # triplets with middle B is (F*F)[2B] // 2.
    #
    # We can obtain the convolution via FFT. We need to choose an FFT size that 
    # is at least high enough to cover indices up to 2*max(S).
    
    max_val = max(S)
    # The maximum index we need is 2*max_val.
    # Find next power of 2 that is >= 2*max_val+1.
    size = 1
    while size < 2 * max_val + 1:
        size *= 2

    # Create the indicator array F.
    F = np.zeros(size, dtype=np.int64)
    for x in S:
        F[x] = 1

    # Convert to float for FFT and perform a real FFT:
    F_float = F.astype(np.float64)
    fftF = np.fft.rfft(F_float)
    conv = np.fft.irfft(fftF * fftF)
    # Round the result to the nearest integer.
    conv = np.rint(conv).astype(np.int64)

    # Now, for every potential middle B in S,
    # if X = 2*B is a valid index, add (conv[2*B] // 2) to answer.
    count = 0
    for b in S:
        idx = 2 * b
        if idx < len(conv):
            count += conv[idx] // 2

    sys.stdout.write(str(count))

if __name__ == '__main__':
    main()