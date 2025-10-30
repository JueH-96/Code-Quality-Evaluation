import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    valid = [False] * N
    for s in range(N - M + 1):
        if S[s:s+M] == T:
            valid[s] = True

    dp = [False] * (N + 1)
    dp[0] = True

    for i in range(1, N + 1):
        dp[i] = False
        start_s = max(0, i - M)
        end_s = min(i - 1, N - M)
        if start_s > end_s:
            continue
        for s in range(start_s, end_s + 1):
            if valid[s] and dp[s]:
                dp[i] = True
                break

    print("Yes" if dp[N] else "No")

if __name__ == "__main__":
    main()