import sys
def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))

    # expire[t] = number of donors expiring right after event t
    expire = [0] * (N + 2)

    D = 0  # D_j = number of donors at event j
    ans = [0] * N

    for j in range(1, N + 1):
        aj = A[j-1]
        # E_j = j + (initial + received stones at j) = j + aj + D
        Ej = j + aj + D

        # if this alien will donate at least once (i.e. C_j0 > 0),
        # then record its expiration at Ej (if within [1..N])
        if Ej > j and Ej <= N:
            expire[Ej] += 1

        # final stones = max(E_j - N, 0)
        if Ej > N:
            ans[j-1] = Ej - N
        # else ans[j-1] stays 0

        # prepare D_{j+1} if j < N
        if j < N:
            # contributed = 1 if this alien starts donating at event j+1
            contributed = 1 if Ej > j else 0
            # remove donors that expire after event j, add this new donor
            D = D - expire[j] + contributed

    # output
    sys.stdout.write(" ".join(map(str, ans)))

if __name__ == "__main__":
    main()