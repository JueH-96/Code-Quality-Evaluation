import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    # Fenwick tree (BIT) for counts of prefix sums mod M
    # Indices 1..M correspond to values 0..M-1 (mapped by idx = value+1)
    bit = [0] * (M + 1)

    # Initialize with P0 = 0
    # Insert one count at index 1 (for prefix sum 0)
    i = 1
    while i <= M:
        bit[i] += 1
        i += i & -i

    sum_old = 0   # sum of P_j for j<r (excluding P0 which is zero)
    ans = 0
    pr = 0        # current prefix sum mod M

    for r in range(1, N + 1):
        ai = int(next(it))
        pr = (pr + ai) % M

        # Query count of previous P_j <= pr
        idx = pr + 1
        cnt_le = 0
        j = idx
        while j > 0:
            cnt_le += bit[j]
            j -= j & -j

        # Number of previous P_j > pr
        cnt_gt = r - cnt_le

        # Contribution for this r
        # sum_{j<r} ((pr - P_j) mod M) = r*pr - sum_old + M * cnt_gt
        ans += r * pr - sum_old + cnt_gt * M

        # Include pr into sum_old for next iterations
        sum_old += pr

        # Update BIT: add one count at index = pr+1
        j = idx
        while j <= M:
            bit[j] += 1
            j += j & -j

    # Print the final answer
    sys.stdout.write(str(ans))


if __name__ == "__main__":
    main()