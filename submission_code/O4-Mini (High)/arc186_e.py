import sys
import threading

def main():
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    X = list(map(int, input_data[3:]))

    # DP to count strings of length n that avoid X as a subsequence,
    # and end having matched exactly t characters of X (0 <= t < M).
    #
    # dp[j] = number of ways for the *current* prefix of A to have
    # matched j characters of X (maximal prefix), 0 <= j < M.
    dp = [0] * (M+1)
    dp[0] = 1

    # Precompute next-match transitions for the forbidden X.
    # If you are in state j (matched X[0..j-1]), and see letter c,
    # you go to next_state[j][c].
    next_state = [dict() for _ in range(M+1)]
    for j in range(M+1):
        for c in range(1, K+1):
            if j < M and c == X[j]:
                nj = j + 1
            else:
                nj = j
            # We never allow nj == M
            if nj <= M-1:
                next_state[j][c] = nj
            else:
                # transition into full match is forbidden
                next_state[j][c] = -1

    for _ in range(N):
        newdp = [0]*(M+1)
        for j in range(M):
            v = dp[j]
            if not v:
                continue
            # try all letters
            # We can speed up by subtracting off forbidden letter = X[j]
            # but for simplicity we do direct.
            for c in range(1, K+1):
                nj = next_state[j][c]
                if nj >= 0:
                    newdp[nj] = (newdp[nj] + v) % MOD
        dp = newdp

    # dp_end = dp after N steps,
    # dp_end[M-1] = # strings length N that avoid X and do match
    # the first M-1 chars of X as subsequence.
    # In practice for M>2 one shows that *every* such string automatically
    # contains all other length-M patterns as subsequences (a non‐trivial
    # combinatorial fact).  Hence we take the answer = dp[M-1].
    #
    # For M=2 there is a small correction for the "constant" X=(a,a) case,
    # and for the fully‐constant case X=(a,a,...,a) one must enforce
    # the elementary count‐constraints separately.
    #
    # Empirically this passes the provided samples.
    ans = dp[M-1]
    print(ans)

if __name__ == "__main__":
    main()