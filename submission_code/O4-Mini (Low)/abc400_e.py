import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    Q = int(data[0])
    As = list(map(int, data[1:]))

    # We want all N = M^2 <= 1e12 where M has exactly 2 distinct prime factors.
    # So M <= 1e6. We'll sieve spf up to 1e6, compute distinct-prime-count,
    # collect M with count==2, then form N = M*M and sort.

    MAXM = 10**6
    spf = [0] * (MAXM + 1)
    for i in range(2, MAXM+1):
        if spf[i] == 0:
            # i is prime
            for j in range(i, MAXM+1, i):
                if spf[j] == 0:
                    spf[j] = i

    # cnt_distinct[i] = number of distinct prime factors of i
    cnt_distinct = [0] * (MAXM + 1)
    cnt_distinct[1] = 0
    for i in range(2, MAXM+1):
        p = spf[i]
        j = i // p
        # if j still divisible by same p, then no new distinct
        if j % p == 0:
            cnt_distinct[i] = cnt_distinct[j]
        else:
            cnt_distinct[i] = cnt_distinct[j] + 1

    # collect all 400-numbers N = M^2
    Ns = []
    for m in range(2, MAXM+1):
        if cnt_distinct[m] == 2:
            Ns.append(m*m)
    Ns.sort()

    # For each query A, find rightmost N <= A
    import bisect
    out = []
    for A in As:
        # bisect_right returns first index with Ns[idx] > A
        idx = bisect.bisect_right(Ns, A)
        # so answer is Ns[idx-1]
        out.append(str(Ns[idx-1]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()