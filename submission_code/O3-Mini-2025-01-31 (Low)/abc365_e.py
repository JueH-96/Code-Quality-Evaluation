def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    
    # Build prefix xor array P such that:
    # P[0] = 0 and P[i] = A_1 ⊕ A_2 ⊕ ... ⊕ A_i for i >= 1.
    P = [0] * (n + 1)
    for i in range(n):
        P[i + 1] = P[i] ^ A[i]

    # Instead of directly summing up the XOR of subarrays,
    # note that the XOR from A_i to A_j is equal to P[i-1] ⊕ P[j] for 1 <= i <= j <= N.
    # Redefine indices: let p[0..n] be our prefix array. Then, we need
    # sum of (p[i] ⊕ p[j]) over all pairs 0 <= i < j <= n.
    # We can solve this by summing up the contribution of each bit independently.
    
    total = 0
    count = n + 1  # total number of prefix values
    # Iterate over possible bit positions
    # We can restrict the bit positions: maximum A_i up to 10^8, so 32 bits is safe.
    for bit in range(32):
        mask = 1 << bit
        ones = 0
        for x in P:
            if x & mask:
                ones += 1
        zeros = count - ones
        # For a given bit, its contribution in a XOR sum is mask (2^bit) if one number has the bit set and the other doesn't.
        # The number of such pairs is ones * zeros.
        total += ones * zeros * mask

    sys.stdout.write(str(total))

if __name__ == "__main__":
    main()