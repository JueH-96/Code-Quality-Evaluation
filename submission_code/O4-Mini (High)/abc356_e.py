import sys
import threading
def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    # read exactly n integers
    A = [int(next(it)) for _ in range(n)]
    M = max(A)
    # frequency array
    freq = [0] * (M + 1)
    for a in A:
        freq[a] += 1
    # prefix sums of frequencies
    csum = [0] * (M + 1)
    s = 0
    for i in range(M + 1):
        s += freq[i]
        csum[i] = s
    ans = 0
    f = freq
    c = csum
    # for each possible smaller value x
    for x in range(1, M + 1):
        fx = f[x]
        if fx == 0:
            continue
        # pairs where both values == x
        if fx > 1:
            ans += fx * (fx - 1) // 2
        # now pairs x < y
        # maximum k such that k*x <= M
        max_k = M // x
        # sum over k = floor(y/x)
        for k in range(1, max_k + 1):
            # compute lower bound L
            if k == 1:
                L = x + 1
            else:
                L = k * x
            # upper bound U
            U = L + x - 1
            if U > M:
                U = M
            if L > U:
                continue
            # sum of freq[y] over y in [L, U]
            G = c[U] - c[L-1]
            # each such pair contributes k
            ans += fx * G * k
    # output the result
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()