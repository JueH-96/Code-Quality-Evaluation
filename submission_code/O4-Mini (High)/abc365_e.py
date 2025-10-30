import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:]))

    # We build prefix XORs P[0..n], where P[0]=0, P[i]=A[1]^...^A[i].
    # The sum over all i<j of XOR(A_i...A_j) equals
    #   sum_{0<=u<v<=n} (P[u] ^ P[v])  minus  sum of the single-element XORs (which is sum(A))
    # The sum over all pairs P[u]^P[v] can be done bitwise:
    #   for each bit b, it contributes 2^b * (#ones in P at b) * (#zeros in P at b).

    # Count how many prefix XORs have bit b = 1
    bit_counts = [0] * 31
    curr = 0
    # P[0]=0 contributes no ones
    for a in A:
        curr ^= a
        # increment count for each bit set in curr
        for b in range(31):
            if (curr >> b) & 1:
                bit_counts[b] += 1

    total_pairs_sum = 0
    total_prefixes = n + 1  # P[0]..P[n]
    for b in range(31):
        cnt1 = bit_counts[b]
        cnt0 = total_prefixes - cnt1
        total_pairs_sum += (1 << b) * cnt1 * cnt0

    # subtract the sum of single-element XORs (i.e., sum(A))
    answer = total_pairs_sum - sum(A)
    print(answer)

if __name__ == "__main__":
    main()