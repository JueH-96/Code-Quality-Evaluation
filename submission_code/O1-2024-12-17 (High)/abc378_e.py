def main():
    import sys
    input = sys.stdin.readline

    # Read inputs
    N, M = map(int, input().split())
    A = list(map(int, input().split()))

    # Compute prefix sums and their remainders mod M
    prefix = [0] * (N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + A[i]
    P = [prefix[i] % M for i in range(N + 1)]  # P[i] = prefix[i] mod M

    # Precompute the sum of P[k] for k in [0..r-1], stored in Sp[r].
    # That is, Sp[r] = P[0] + P[1] + ... + P[r-1].
    Sp = [0] * (N + 1)
    for r in range(1, N + 1):
        Sp[r] = Sp[r - 1] + P[r - 1]

    # Fenwick (Binary Indexed) Tree for counting frequencies of remainders.
    fenw = [0] * (M + 1)

    def fenw_update(i, v):
        # Updates Fenwicks for frequency of remainder
        while i <= M:
            fenw[i] += v
            i += i & -i

    def fenw_sum(i):
        # Returns the count of all remainders <= (i - 1)
        s = 0
        while i > 0:
            s += fenw[i]
            i -= i & -i
        return s

    # Insert P[0] into Fenwicks (this accounts for subarrays starting at index 1).
    fenw_update(P[0] + 1, 1)

    ans = 0

    # For each r in [1..N], calculate the contribution of subarrays ending at r.
    for r in range(1, N + 1):
        # # of prefix indices k in [0..r-1] with P[k] > P[r] is:
        #   biggerCount = r - (# of prefix indices with P[k] <= P[r])
        biggerCount = r - fenw_sum(P[r] + 1)

        # Contribution from all subarrays that end at r:
        # f(r) = sum_{k=0..r-1} ( (P[r] - P[k]) mod M )
        #      = r*P[r] - sum_{k=0..r-1} P[k] + M * biggerCount
        contrib = r * P[r] - Sp[r] + M * biggerCount
        ans += contrib

        # Now add P[r] to Fenwicks for next iterations
        fenw_update(P[r] + 1, 1)

    print(ans)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()