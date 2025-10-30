import sys

def main() -> None:
    # Read input
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # Build prefix–xor array P[0..N]
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = P[i - 1] ^ A[i - 1]

    # Maximum bit we have to inspect (A_i ≤ 10^8 < 2^27)
    MAX_BIT = 31        # 0 … 30 is plenty

    ans = 0
    # For every bit position deal with parities independently
    for b in range(MAX_BIT):
        # Count pairs (l, r) with l < r and P_l ⊕ P_r has bit b = 1
        cnt = [0, 0]               # how many prefixes with parity 0/1 seen so far
        total_diff = 0             # pairs with differing parity (distance ≥ 1)
        for val in P:
            parity = (val >> b) & 1
            total_diff += cnt[1 ^ parity]   # earlier prefixes with opposite parity
            cnt[parity] += 1

        # Sub-arrays of length 1 correspond to adjacent prefix positions (r = l + 1).
        # Remove those pairs because they are not included in the problem statement.
        adjacent_diff = 0
        for i in range(N):
            if ((P[i] ^ P[i + 1]) >> b) & 1:
                adjacent_diff += 1

        valid_pairs = total_diff - adjacent_diff
        ans += valid_pairs << b     # contribution of this bit

    print(ans)

if __name__ == "__main__":
    main()