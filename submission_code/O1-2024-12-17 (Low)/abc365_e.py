def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute prefix XOR array B of length N+1
    # B[i] = A[0] XOR A[1] XOR ... XOR A[i-1], with B[0] = 0
    B = [0] * (N + 1)
    for i in range(1, N + 1):
        B[i] = B[i - 1] ^ A[i - 1]

    # 1) f(B) = sum of XOR over all pairs (x < y) from B
    # To get this efficiently, do it bit by bit.
    # For each bit k, count how many elements in B have that bit set.
    # If cnt_k is that count, the number of pairs that differ in that bit
    # is cnt_k * (len(B) - cnt_k), and each such pair contributes 2^k.
    total_pairs_xor = 0
    length = N + 1
    for k in range(31):  # 31 bits is enough since A_i up to 10^8
        mask = 1 << k
        cnt_k = sum(1 for x in B if x & mask)
        total_pairs_xor += (cnt_k * (length - cnt_k)) << k

    # 2) Subtract the consecutive XORs: sum of B[x] ^ B[x+1] for x = 0..N-1
    consecutive_xor_sum = 0
    for x in range(N):
        consecutive_xor_sum += B[x] ^ B[x + 1]

    # The answer is f(B) - sum of consecutive XORs
    answer = total_pairs_xor - consecutive_xor_sum

    print(answer)

# Do not forget to call main()
if __name__ == "__main__":
    main()