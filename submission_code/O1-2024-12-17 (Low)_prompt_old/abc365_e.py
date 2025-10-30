def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute prefix XOR array P of length N+1, with P[0] = 0
    P = [0] * (N + 1)
    for i in range(N):
        P[i+1] = P[i] ^ A[i]

    # We will compute the sum over all pairs (i<j) of (P[j] ^ P[i]),
    # i.e. subarrays of length >=1, then subtract the sum of single elements
    # (which corresponds to subarrays of length=1).
    #
    # Let S1 = sum_{0 <= i < j <= N} (P[j] ^ P[i]).
    # The final answer we want = S1 - sum(A).

    # To compute S1 efficiently, note that summing XORs bit by bit is easier.
    # For a particular bit k, let c1 be the number of prefix XORs whose k-th bit is 1.
    # Let c0 = (N+1) - c1 be the number whose k-th bit is 0.
    # Then the number of pairs (i<j) for which the k-th bit of (P[j] ^ P[i]) is 1 is c0 * c1.
    # Hence the total contribution from bit k is (c0 * c1) * (1 << k).

    count_bit = [0]*32  # count of how many prefix XORs have each bit set
    for val in P:
        # Count bits set in val
        b = val
        idx = 0
        while b > 0:
            if (b & 1) == 1:
                count_bit[idx] += 1
            b >>= 1
            idx += 1
        # If val has fewer than 32 bits, the remaining bits stay 0

    S1 = 0
    total_prefixes = N + 1
    for k in range(32):
        c1 = count_bit[k]
        c0 = total_prefixes - c1
        S1 += (c0 * c1) << k

    # Subtract the sum of A to remove the subarrays of length=1
    answer = S1 - sum(A)

    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()