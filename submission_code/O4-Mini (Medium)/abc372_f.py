import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))
    X = [0] * M
    Y = [0] * M
    for i in range(M):
        # convert to 0-based
        xi = int(next(it)) - 1
        yi = int(next(it)) - 1
        X[i] = xi
        Y[i] = yi

    mod = 998244353

    # f[j] = dp_k[(j + k) % N]
    f = [0] * N
    f[0] = 1  # dp_0[0] = 1

    # iterate k = 0 .. K-1 to build f -> f_{k+1}
    # we only need f array, we accumulate extra-edge contributions each step
    Nval = N  # local copy
    for k in range(K):
        # compute k mod N and (k+1) mod N
        km = k % Nval
        k1m = km + 1
        if k1m == Nval:
            k1m = 0
        # collect updates from extra edges
        upd = []
        # local references
        f_loc = f
        X_loc = X
        Y_loc = Y
        for t in range(M):
            # source index in f_k:
            p = X_loc[t] - km
            if p < 0:
                p += Nval
            # target index in f_{k+1}:
            i = Y_loc[t] - k1m
            if i < 0:
                i += Nval
            val = f_loc[p]
            if val:
                upd.append((i, val))
        # apply updates
        for (i, val) in upd:
            f[i] = (f[i] + val) % mod

    # after K steps, f holds f_K
    # answer is dp_K[0] = f_K[(0 - K) mod N]
    km = K % N
    idx = -km
    if idx < 0:
        idx += N
    ans = f[idx] % mod
    print(ans)

if __name__ == "__main__":
    main()