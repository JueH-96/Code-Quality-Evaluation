import sys

# ---------- Fast Walsh–Hadamard Transform for XOR ----------
def fwht(a):
    """In place FWHT (XOR), size must be power of two"""
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h << 1):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h <<= 1


# ---------- main ----------
def main() -> None:
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    H = int(next(it))
    W = int(next(it))
    N = 1 << W                       # 2**W

    # frequency of every row mask
    freq = [0] * N
    for _ in range(H):
        s = next(it)
        m = 0
        for j, ch in enumerate(s):
            if ch == '1':
                m |= 1 << j
        freq[m] += 1

    # F[mask] = min(popcount(mask), W - popcount(mask))
    F = [0] * N
    for m in range(N):
        k = m.bit_count()
        F[m] = k if k <= W - k else W - k

    # FWHT
    freq_hat = freq[:]          # copy, will be transformed in place
    F_hat = F[:]

    fwht(freq_hat)
    fwht(F_hat)

    # point-wise multiplication
    G_hat = [freq_hat[i] * F_hat[i] for i in range(N)]

    # inverse FWHT  (same transform again)
    fwht(G_hat)
    inv_scale = N               # because FWHT^2 = N * Id
    res = [x // inv_scale for x in G_hat]

    # answer
    print(min(res))


# -----------------------------------------------------------
if __name__ == "__main__":
    main()