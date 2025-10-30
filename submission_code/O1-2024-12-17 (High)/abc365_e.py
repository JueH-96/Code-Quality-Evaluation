def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Build prefix-xor array P of length N+1
    # P[i] = A[0] ^ A[1] ^ ... ^ A[i-1]
    # P[0] = 0 by definition
    P = [0] * (N + 1)
    for i in range(N):
        P[i + 1] = P[i] ^ A[i]

    # We want:
    #   S = Σ (A_i ^ A_{i+1} ^ ... ^ A_j)  for 1 ≤ i < j ≤ N
    #
    # Observe that A_i ^ A_{i+1} ^ ... ^ A_j = P[i-1] ^ P[j].
    # So we consider all pairs (x, y) = (i-1, j) where x+1 < y+1 ⇒ y ≥ x+2.
    #
    # A simpler way is to compute:
    #   1) S1 = Σ(P[x] ^ P[y]) over all 0 ≤ x < y ≤ N
    #   2) S2 = Σ(P[x] ^ P[x+1]) for x = 0..N-1  (these correspond to subarrays of length 1
    #      when mapped back to i, j = i ⇒ we don't want them)
    #
    # Then the final answer = S1 - S2.

    # Compute S2 in O(N)
    S2 = 0
    for x in range(N):
        S2 += P[x] ^ P[x + 1]

    # Compute S1 by counting, bit by bit.
    #
    # For each bit b, let mask = (1 << b). We'll do a single pass from right to left:
    #   - Maintain how many prefix-xors we've seen so far (on the right) that have bit b = 0 or 1.
    #   - For the current index x, if P[x] has bit b = 1, it forms a pair with all future indices 
    #     whose bit b = 0. If P[x] has bit b = 0, it pairs with all future indices with bit b = 1.
    #
    # This effectively counts how many pairs (x < y) set that bit. Then we multiply by 2^b
    # and sum over b.

    M = N + 1
    S1 = 0
    for b in range(31):  # up to 2^30 > 10^8
        mask = 1 << b
        c0 = 0
        c1 = 0
        bit_pairs = 0
        # Traverse from right to left
        for x in range(M - 1, -1, -1):
            if P[x] & mask:
                bit_pairs += c0  # pairs with those that have bit b = 0
                c1 += 1
            else:
                bit_pairs += c1  # pairs with those that have bit b = 1
                c0 += 1
        S1 += bit_pairs * mask

    # Final answer
    print(S1 - S2)

# Do not forget to call main()
main()