# YOUR CODE HERE
import sys
MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())
    conditions = []
    for _ in range(M):
        L, R, X = map(int, sys.stdin.readline().split())
        conditions.append((L-1, R, X-1))  # converting to 0-based index

    # Initialize dp array
    dp = [0] * (N + 1)
    dp[0] = 1

    for i in range(N):
        new_dp = [0] * (N + 1)
        for j in range(N + 1):
            if dp[j] == 0:
                continue
            # Try placing the (i+1)-th element in position j
            # Check all conditions that involve position j
            valid = True
            for L, R, X in conditions:
                if L <= j < R and X == j:
                    valid = False
                    break
            if valid:
                new_dp[j] = (new_dp[j] + dp[j]) % MOD
            # Try placing the (i+1)-th element in other positions
            for k in range(N):
                if k == j:
                    continue
                # Check all conditions that involve position k
                valid = True
                for L, R, X in conditions:
                    if L <= k < R and X == k:
                        valid = False
                        break
                if valid:
                    new_dp[k] = (new_dp[k] + dp[j]) % MOD
        dp = new_dp

    total = sum(dp) % MOD
    print(total)

if __name__ == "__main__":
    main()