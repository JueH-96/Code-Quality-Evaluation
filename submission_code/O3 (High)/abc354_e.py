import sys
sys.setrecursionlimit(1 << 25)

def main() -> None:
    import sys

    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))

    A = []
    B = []
    for _ in range(N):
        A.append(int(next(it)))
        B.append(int(next(it)))

    # Pre–compute every pair of indices (i, j) whose front numbers
    # or back numbers are the same.  Only these pairs are legal moves.
    legal_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if A[i] == A[j] or B[i] == B[j]:
                legal_pairs.append((i, j))

    FULL = (1 << N) - 1
    memo = [-1] * (1 << N)        # ‑1 = unknown, 0 = losing, 1 = winning

    def dfs(mask: int) -> int:
        """Return 1 if the current player has a winning strategy
        starting from the given mask, 0 otherwise."""
        if memo[mask] != -1:
            return memo[mask]

        # Try every legal pair that is still on the table.
        for i, j in legal_pairs:
            if (mask >> i) & 1 and (mask >> j) & 1:        # both cards present
                nxt = mask ^ (1 << i) ^ (1 << j)           # remove them
                if dfs(nxt) == 0:                          # opponent loses
                    memo[mask] = 1
                    return 1

        # No move, or every move lets opponent win → losing state
        memo[mask] = 0
        return 0

    winner = dfs(FULL)
    print("Takahashi" if winner else "Aoki")

if __name__ == "__main__":
    main()