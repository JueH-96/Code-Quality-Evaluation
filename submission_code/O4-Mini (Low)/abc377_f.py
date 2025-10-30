import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    it = 2
    R = set()
    C = set()
    D1 = set()  # i - j
    D2 = set()  # i + j
    for _ in range(M):
        a = int(data[it]); b = int(data[it+1]); it += 2
        R.add(a)
        C.add(b)
        D1.add(a - b)
        D2.add(a + b)
    # Convert to sorted lists for bisect
    import bisect
    Rl = sorted(R)
    Cl = sorted(C)
    D1l = list(D1)
    D2l = list(D2)
    lenR = len(Rl)
    lenC = len(Cl)

    # 1st order sums
    SA = N * lenR
    SB = N * lenC
    SC = 0
    for d in D1l:
        SC += N - abs(d)
    SD = 0
    # for s in [2..2N], count of (i,j) with i+j = s
    for s in D2l:
        if 2 <= s <= N+1:
            SD += (s - 1)
        elif N+1 < s <= 2*N:
            SD += (2*N + 1 - s)
        # else zero

    # 2nd order intersections
    AB = lenR * lenC

    # A ∩ C: i in R AND (i-j)=d for some d => j=i-d in [1,N]
    AC = 0
    for d in D1l:
        # i ∈ R ∩ [1+d, N+d]
        lo = 1 + d
        hi = N + d
        # count Rl entries in [lo,hi]
        li = bisect.bisect_left(Rl, lo)
        ri = bisect.bisect_right(Rl, hi)
        AC += (ri - li)

    # A ∩ D2: i in R AND (i+j)=s => j=s-i in [1,N] => i∈[s-N, s-1]
    AD = 0
    for s in D2l:
        lo = s - N
        hi = s - 1
        li = bisect.bisect_left(Rl, lo)
        ri = bisect.bisect_right(Rl, hi)
        AD += (ri - li)

    # B ∩ C: j in C AND (i-j)=d => i=j+d in [1,N] => j∈[1-d, N-d]
    BC = 0
    for d in D1l:
        lo = 1 - d
        hi = N - d
        li = bisect.bisect_left(Cl, lo)
        ri = bisect.bisect_right(Cl, hi)
        BC += (ri - li)

    # B ∩ D2: j in C AND (i+j)=s => i=s-j in [1,N] => j∈[s-N, s-1]
    BD = 0
    for s in D2l:
        lo = s - N
        hi = s - 1
        li = bisect.bisect_left(Cl, lo)
        ri = bisect.bisect_right(Cl, hi)
        BD += (ri - li)

    # C ∩ D2: solve i-j=d and i+j=s => i=(d+s)/2, j=(s-d)/2 must be integer and 1<=i,j<=N
    CD = 0
    for d in D1l:
        for s in D2l:
            tmp = d + s
            if tmp & 1:
                continue
            i = tmp // 2
            j = (s - d) // 2
            if 1 <= i <= N and 1 <= j <= N:
                CD += 1

    # 3rd order
    # A ∩ B ∩ C: (i,j) ∈ R×C with i-j ∈ D1
    ABC = 0
    for i in Rl:
        for d in D1l:
            j = i - d
            # check j in C?
            # we do a binary search
            # or use direct set membership
            if 1 <= j <= N and j in C:
                ABC += 1

    # A ∩ B ∩ D2: (i,j) ∈ R×C with i+j ∈ D2
    ABD = 0
    for i in Rl:
        for s in D2l:
            j = s - i
            if 1 <= j <= N and j in C:
                ABD += 1

    # A ∩ C ∩ D2: i in R, and it lies on diag1 d and diag2 s => unique (i,j)
    ACD = 0
    for d in D1l:
        for s in D2l:
            tmp = d + s
            if tmp & 1:
                continue
            i = tmp // 2
            j = (s - d) // 2
            if 1 <= i <= N and 1 <= j <= N and i in R:
                ACD += 1

    # B ∩ C ∩ D2: j in C and on both diags
    BCD = 0
    for d in D1l:
        for s in D2l:
            tmp = d + s
            if tmp & 1:
                continue
            i = tmp // 2
            j = (s - d) // 2
            if 1 <= i <= N and 1 <= j <= N and j in C:
                BCD += 1

    # 4th order: A ∩ B ∩ C ∩ D2: (i,j) ∈ R×C with both diag conditions
    ABCD = 0
    for i in Rl:
        for j in Cl:
            if (i - j) in D1 and (i + j) in D2:
                ABCD += 1

    # Inclusion–exclusion
    union = (SA + SB + SC + SD
             - (AB + AC + AD + BC + BD + CD)
             + (ABC + ABD + ACD + BCD)
             - ABCD)

    # safe = total - attacked
    ans = N * N - union
    print(ans)


if __name__ == "__main__":
    main()