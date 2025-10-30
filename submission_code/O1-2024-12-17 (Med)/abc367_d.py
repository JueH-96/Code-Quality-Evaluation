def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    # Compute prefix sums mod M: prefix[i] = (A[0] + ... + A[i-1]) mod M
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i+1] = (prefix[i] + A[i]) % M

    # p[i] = prefix[i+1], for i in [0..N-1]
    p = [prefix[i+1] for i in range(N)]

    # Count how many indices share the same remainder for the "t > s" case
    freq = [0] * M
    for x in p:
        freq[x] += 1

    # Number of pairs (s,t) with s < t and p[s] = p[t]
    c1 = 0
    for c in freq:
        if c > 1:
            c1 += c * (c - 1) // 2

    # For the "t < s" case, we need p[t] = p[s] + offset (mod M)
    # offset = (M - totalSumMod) mod M
    totalSumMod = prefix[N]  # same as p[N-1] if N>0
    offset = (M - totalSumMod) % M

    # We'll count how many pairs (i, j) with i>j such that p[j] = p[i] + offset
    # Equivalently, p[j] - offset = p[i].  Define X[j] = (p[j] - offset) mod M.
    # Then X[j] = p[i].  We iterate i from left to right, counting how many j < i
    # satisfy X[j] = p[i].
    freqX = [0] * M
    c2 = 0
    for i in range(N):
        c2 += freqX[p[i]]  # count j < i with X[j] = p[i]
        x = p[i] - offset
        if x < 0:
            x += M
        freqX[x] += 1

    print(c1 + c2)

# Do not forget to call main at the end!
main()