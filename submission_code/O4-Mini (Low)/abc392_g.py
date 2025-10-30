import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    S = list(map(int, data[1:]))

    maxv = max(S)
    size = 1
    # need convolution up to index 2*maxv
    while size <= 2 * maxv:
        size <<= 1

    # build array of complex numbers for FFT
    A = [0] * size
    for x in S:
        A[x] = 1.0

    # iterative FFT
    def fft(a, invert):
        n = len(a)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j |= bit
            if i < j:
                a[i], a[j] = a[j], a[i]
        length = 2
        while length <= n:
            ang = 2 * 3.141592653589793 / length * (-1 if invert else 1)
            wlen = complex(math.cos(ang), math.sin(ang))
            for i in range(0, n, length):
                w = 1+0j
                half = length >> 1
                for j in range(half):
                    u = a[i + j]
                    v = a[i + j + half] * w
                    a[i + j] = u + v
                    a[i + j + half] = u - v
                    w *= wlen
            length <<= 1
        if invert:
            for i in range(n):
                a[i] /= n

    import math
    # convert A to complex
    A = list(map(complex, A))
    fft(A, False)
    for i in range(size):
        A[i] *= A[i]
    fft(A, True)

    # now A[k].real is conv[k], rounded
    ans = 0
    # For each B in S, conv[2B] = 2 * count + 1
    for B in S:
        c = int(round(A[2 * B].real))
        # subtract the (B,B) pair and divide by 2
        ans += (c - 1) // 2

    print(ans)

if __name__ == "__main__":
    main()