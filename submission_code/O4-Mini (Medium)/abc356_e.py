import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Frequency array up to max value
    MAXV = max(A)
    freq = [0] * (MAXV + 1)
    for v in A:
        freq[v] += 1

    # Prefix sums of freq for range queries
    pf = [0] * (MAXV + 1)
    acc = 0
    for i in range(MAXV + 1):
        acc += freq[i]
        pf[i] = acc

    # Unique values in A
    uniq = [i for i in range(1, MAXV + 1) if freq[i] > 0]

    res = 0
    # For each possible minimum value s
    for s in uniq:
        fs = freq[s]
        # Pairs of equal elements (floor(s/s) == 1)
        if fs > 1:
            res += fs * (fs - 1) // 2

        # k = 1 separately: handle L in (s, 2s-1]
        L = s + 1
        if L <= MAXV:
            R = 2 * s - 1
            if R > MAXV:
                R = MAXV
            if L <= R:
                tot1 = pf[R] - pf[L - 1]
                # each such pair contributes floor(l/s) = 1
                res += fs * tot1

        # k >= 2
        # for each k, values l in [k*s, (k+1)*s - 1]
        maxk = MAXV // s
        for k in range(2, maxk + 1):
            L = k * s
            if L > MAXV:
                break
            R = (k + 1) * s - 1
            if R > MAXV:
                R = MAXV
            cnt = pf[R] - pf[L - 1]
            if cnt:
                res += fs * cnt * k

    # Output the result
    sys.stdout.write(str(res))

if __name__ == "__main__":
    main()