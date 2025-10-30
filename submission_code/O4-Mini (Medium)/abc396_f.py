import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # Prepare lists of positions for each value v in [0, M)
    # positions are 1-based
    cnt = [0] * M
    sumpos = [0] * M
    for idx, v in enumerate(A, start=1):
        cnt[v] += 1
        sumpos[v] += idx
    # Compute initial inversion count of A
    # BIT over values 0..M-1
    size = M + 5
    bit = [0] * size
    def bit_update(i, v):
        # i: 1-based index in BIT
        while i < size:
            bit[i] += v
            i += i & -i
    def bit_query(i):
        # sum from 1..i inclusive
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    inv = 0
    # traverse A, for each, count how many previous > A_i
    # we maintain count of seen values in BIT
    # map value v in [0,M) to BIT index v+1
    seen = 0
    for v in A:
        # number of seen values <= v
        le = bit_query(v+1)
        # number of seen > v
        gt = seen - le
        inv += gt
        seen += 1
        bit_update(v+1, 1)

    out = []
    # iterate k from 0..M-1
    # for current inv corresponds to k
    # after printing, compute delta to go to k+1
    # x = (M-1-k) % M
    # C = cnt[x], S = sumpos[x]
    # delta = 2*S - C*(N+1)
    for k in range(M):
        out.append(str(inv))
        x = (M - 1 - k) % M
        C = cnt[x]
        if C:
            S = sumpos[x]
            # compute delta
            delta = 2 * S - C * (N + 1)
            inv += delta
        # if C==0, delta is zero, inv unchanged

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()