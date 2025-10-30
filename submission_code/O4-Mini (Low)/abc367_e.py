def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    X = [int(next(it)) - 1 for _ in range(N)]
    A = [int(next(it)) for _ in range(N)]

    # Precompute binary lifting table up[j][i] = (2^j)-th ancestor of i under mapping X
    import math
    LOG = K.bit_length()  # enough bits to cover K up to 10^18
    up = [X]
    for j in range(1, LOG):
        prev = up[j - 1]
        curr = [0] * N
        for i in range(N):
            curr[i] = prev[prev[i]]
        up.append(curr)

    # For each i, find the node reached by following K steps from i
    res = [0] * N
    for i in range(N):
        cur = i
        k = K
        b = 0
        while k:
            if k & 1:
                cur = up[b][cur]
            k >>= 1
            b += 1
        # After K steps from i we land at cur, so the new A_i = old A[cur]
        res[i] = A[cur]

    # Output
    out = sys.stdout
    out.write(" ".join(map(str, res)))

if __name__ == "__main__":
    main()