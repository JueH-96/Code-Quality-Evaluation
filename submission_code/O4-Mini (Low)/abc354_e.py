import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = [0]*N
    B = [0]*N
    for i in range(N):
        A[i] = int(next(it))
        B[i] = int(next(it))
    # Precompute valid pairs
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            if A[i] == A[j] or B[i] == B[j]:
                pairs.append((i, j))
    FULL = (1 << N) - 1
    dp = [False] * (1 << N)
    # dp[mask] = True if current player to move wins with remaining cards = mask
    # dp[0] = False (no move)
    for mask in range(1, FULL + 1):
        # Try every removable pair
        win = False
        for i, j in pairs:
            bit_i = 1 << i
            bit_j = 1 << j
            if (mask & bit_i) and (mask & bit_j):
                nxt = mask ^ bit_i ^ bit_j
                if not dp[nxt]:
                    win = True
                    break
        dp[mask] = win
    # Full initial mask
    if dp[FULL]:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()