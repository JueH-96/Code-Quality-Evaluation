import sys
sys.setrecursionlimit(10**7)

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    A = [0]*N
    B = [0]*N
    for i in range(N):
        a, b = map(int, input().split())
        A[i], B[i] = a, b

    # Precompute all pairs of cards that can be removed together
    # Represent cards i,j by their bit masks (1<<i, 1<<j)
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if A[i] == A[j] or B[i] == B[j]:
                pairs.append((1 << i, 1 << j))

    FULL = (1 << N) - 1
    dp = [-1] * (1 << N)  # -1 = unknown, 0 = losing, 1 = winning

    def dfs(mask):
        # If we've computed this state, return it
        if dp[mask] != -1:
            return dp[mask]
        # Try every valid removal pair
        for bi, bj in pairs:
            if (mask & bi) and (mask & bj):
                nxt = mask ^ bi ^ bj
                # If the opponent loses on nxt, we win here
                if dfs(nxt) == 0:
                    dp[mask] = 1
                    return 1
        # No winning move found => losing state
        dp[mask] = 0
        return 0

    # Base case: empty mask or any mask with <2 cards is losing
    dp[0] = 0
    if N == 1:
        dp[1] = 0

    # Evaluate from the full set of cards
    winner = dfs(FULL)
    print("Takahashi" if winner == 1 else "Aoki")

if __name__ == "__main__":
    main()