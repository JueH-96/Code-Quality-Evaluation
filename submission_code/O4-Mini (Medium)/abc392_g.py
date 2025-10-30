import sys

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    S = list(map(int, (next(it) for _ in range(n))))
    if n == 0:
        print(0)
        return

    # We will compute the convolution f * f via FFT using numpy.
    # f[x] = 1 if x in S, else 0. Then for each B in S,
    # the number of ordered pairs (A,C) with A+C=2B is (f*f)[2B].
    # Subtract the trivial A=C=B term, divide by 2 to get the count
    # of fine triplets centered at B: ( (f*f)[2B] - 1 ) // 2.
    try:
        import numpy as np
    except ImportError:
        # Fallback for tiny n: direct check (will time out if n large,
        # but in environments without numpy n is usually small).
        sset = set(S)
        S.sort()
        cnt = 0
        for i in range(n):
            A = S[i]
            for j in range(i+1, n):
                B = S[j]
                C = 2*B - A
                if C > B and C in sset:
                    cnt += 1
        print(cnt)
        return

    max_s = max(S)
    # next power of two >= 2*max_s+1
    size = 1
    while size < 2*max_s + 1:
        size <<= 1

    # build f
    f = np.zeros(size, dtype=np.int8)
    f[np.array(S, dtype=np.int32)] = 1

    # FFT convolution
    # rfft gives size//2+1 complex numbers
    F = np.fft.rfft(f)
    # elementwise square
    G = F * F
    # inverse transform back to real, length=size
    conv = np.fft.irfft(G, n=size)
    # round to nearest ints
    conv = np.rint(conv).astype(np.int64)

    total = 0
    for b in S:
        t = conv[2*b] - 1       # subtract the (B,B) trivial pair
        if t > 0:
            total += t // 2

    print(int(total))

if __name__ == "__main__":
    main()