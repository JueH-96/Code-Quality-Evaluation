import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    mod = 998244353
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    B = [0]*(N+1)
    q_total = 0
    for i in range(1, N+1):
        b = int(next(it))
        B[i] = b
        if b == -1:
            q_total += 1

    # precompute M^k
    powM = [1]*(q_total+1)
    for k in range(1, q_total+1):
        powM[k] = powM[k-1] * M % mod

    # base[v] = (M-v)^q_pre for current i
    # v in 1..M
    base = [1] * (M+1)

    # remaining minus ones from current i..N
    rem = q_total

    ans = powM[q_total]  # i = 1 contribution
    # process i=1 initial update
    # update min fixed prefix and rem
    min_fixed = M+1
    if B[1] == -1:
        rem -= 1
    else:
        # fixed
        min_fixed = B[1]

    # iterate i = 2..N
    for i in range(2, N+1):
        # update base by previous position (i-1)
        if B[i-1] == -1:
            # multiply base[v] *= (M-v)
            for v in range(1, M+1):
                base[v] = base[v] * (M - v) % mod

        # compute q_pre and q_post
        q_pre = q_total - rem
        # q_post: rem minus if current is -1
        if B[i] == -1:
            q_post = rem - 1
        else:
            q_post = rem

        mul_post = powM[q_post]

        # contribution for position i
        if B[i] == -1:
            # sum over v = 1..min_fixed-1
            lim = min_fixed - 1
            s = 0
            # if lim < 1, none
            if lim >= 1:
                # sum base[v]
                bv = base  # alias
                for v in range(1, lim+1):
                    s += bv[v]
                    if s >= mod: s -= mod
            ans = (ans + s * mul_post) % mod
        else:
            x = B[i]
            if x < min_fixed:
                # allowed single v = x
                s = base[x]
                ans = (ans + s * mul_post) % mod

        # now update prefix stats with position i
        if B[i] == -1:
            rem -= 1
        else:
            if B[i] < min_fixed:
                min_fixed = B[i]

    print(ans % mod)

if __name__ == "__main__":
    main()