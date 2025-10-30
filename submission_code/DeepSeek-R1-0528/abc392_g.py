import math
import cmath
import sys

def fft(x, inverse=False):
    n = len(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    m = 1
    while m < n:
        angle = 2 * cmath.pi / (2 * m)
        if inverse:
            angle = -angle
        wlen = cmath.exp(1j * angle)
        for i in range(0, n, 2 * m):
            w = 1.0
            for k in range(i, i + m):
                t = x[k + m] * w
                x[k + m] = x[k] - t
                x[k] = x[k] + t
                w *= wlen
        m <<= 1
    if inverse:
        for i in range(n):
            x[i] /= n

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    n = int(data[0])
    S = list(map(int, data[1:1 + n]))
    if n == 0:
        print(0)
        return
    max_val = max(S)
    size_f = 2 * max_val + 1
    f = [0] * size_f
    for num in S:
        if num < size_f:
            f[num] = 1
    L = size_f
    nfft = 1
    while nfft < 2 * L - 1:
        nfft *= 2
    fa = [complex(x, 0) for x in f] + [complex(0, 0)] * (nfft - L)
    fft(fa)
    for i in range(nfft):
        fa[i] = fa[i] * fa[i]
    fft(fa, inverse=True)
    conv = [0] * (2 * max_val + 1)
    for k in range(len(conv)):
        conv_val = fa[k].real
        conv[k] = round(conv_val)
    count = 0
    for b in S:
        idx = 2 * b
        if idx <= 2 * max_val:
            c_val = conv[idx]
            count += (c_val - 1) // 2
    print(count)

if __name__ == '__main__':
    main()