import sys
import cmath

def fft(x, inv=False):
    n = len(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j &= ~bit
            bit >>= 1
        j |= bit
        if i < j:
            x[i], x[j] = x[j], x[i]
    m = 2
    while m <= n:
        wlen = cmath.exp(2j * cmath.pi / m * (-1 if inv else 1))
        for k in range(0, n, m):
            w = 1
            for j in range(m // 2):
                idx1 = k + j
                idx2 = k + j + (m // 2)
                u = x[idx1]
                v = x[idx2] * w
                x[idx1] = u + v
                x[idx2] = u - v
                w *= wlen
        m <<= 1
    if inv:
        for i in range(n):
            x[i] /= n

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    S = list(map(int, data[1:1+n]))
    M = max(S)
    U = 2 * M
    conv_size = 2 * U + 1
    n_len = 1
    while n_len < conv_size:
        n_len <<= 1

    arr = [0] * (U + 1)
    for num in S:
        if num <= U:
            arr[num] = 1

    a_pad = arr + [0] * (n_len - len(arr))
    fft(a_pad)
    c_fft = [x * x for x in a_pad]
    fft(c_fft, inv=True)
    conv = [round(x.real) for x in c_fft]

    count = 0
    for b in S:
        k = 2 * b
        if k < n_len:
            mid = k // 2
            diag = 1 if (mid <= U and arr[mid] == 1) else 0
            distinct_pairs = (conv[k] - diag) // 2
            count += distinct_pairs
    print(count)

if __name__ == '__main__':
    main()