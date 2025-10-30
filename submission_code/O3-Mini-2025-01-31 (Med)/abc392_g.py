def main():
    import sys, math, numpy as np
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    # read the set S; since the elements are distinct, we simply mark them in an indicator array
    s_list = list(map(int, data[1:]))
    if n < 3:
        sys.stdout.write("0")
        return

    # Find the maximum value in S to size our indicator array.
    max_val = max(s_list)
    size = max_val + 1  # we need indices 0...max_val
    # Create indicator array f where f[x] = 1 if x in S, else 0.
    # We also retain the original S in s_list for iterating later.
    f = np.zeros(size, dtype=np.int64)
    for x in s_list:
        f[x] = 1

    # Explanation:
    # A triplet (A, B, C) is "fine" if B - A = C - B, which implies 2 * B = A + C.
    # Notice that if we select A and C from S with A < C and (A+C) is even and 
    # B = (A+C)//2 is in S, then automatically A < B < C.
    #
    # To efficiently count such pairs (A, C) for each potential middle B in S, 
    # we can use convolution. In particular, if we let f be the indicator array, then
    # the convolution (f * f)[t] equals the number of ordered pairs (A, C) with A + C = t.
    # For a valid triplet with middle element B, we must have t = 2 * B and the pair (A, C)
    # must be distinct. The convolution will count the pair (B, B) (if B in S), so we subtract 1.
    # Then, each unordered pair is counted twice (as (A, C) and (C, A)); hence, we divide by 2.
    #
    # Therefore the number of triplets using B (which equals x in S) is 
    #      (conv[2*x] - 1) // 2, provided 2*x lies in the convolution result.
    
    # We can perform the convolution using FFT. The length L for zero-padding must be at least:
    #      L >= 2*size - 1.
    L = 1
    target = 2 * size - 1
    while L < target:
        L *= 2

    # Convert f to float64 for FFT; then compute the FFT, multiply element‐wise, 
    # and use the inverse FFT. The result is our convolution.
    f_float = f.astype(np.float64)
    F = np.fft.rfft(f_float, n=L)
    conv = np.fft.irfft(F * F, n=L)
    # Round the results to the nearest integer as slight floating point errors may occur.
    conv = np.rint(conv).astype(np.int64)

    # Now count fine triplets.
    total = 0
    # For each x in S, consider it as a potential middle element B.
    # The required sum A + C = 2 * B corresponds to index 2*x in the convolution result.
    for x in s_list:
        idx = 2 * x
        if idx < len(conv):
            # conv[idx] counts all ordered pairs (A, C) with sum 2*x.
            # Subtract 1 for the pair (x, x) and then divide by 2 to count unordered distinct pairs.
            count_pairs = conv[idx] - 1
            if count_pairs > 0:
                total += count_pairs // 2

    sys.stdout.write(str(total))


if __name__ == '__main__':
    main()