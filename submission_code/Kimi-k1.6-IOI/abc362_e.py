import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    ans = [0] * (N + 1)
    ans[1] = N % MOD
    dp = [ defaultdict(lambda: defaultdict(int)) for _ in range(N) ]
    for i in range(N):
        current_dp = defaultdict(lambda: defaultdict(int))
        for j in range(i):
            d = A[i] - A[j]
            for l in dp[j][d]:
                cnt = dp[j][d][l]
                current_dp[d][l+1] += cnt
            current_dp[d][2] += 1
        for d in current_dp:
            for l in current_dp[d]:
                ans[l] = (ans[l] + current_dp[d][l]) % MOD
        dp[i] = current_dp
    print(' '.join(map(str, ans[1:N+1])))

if __name__ == '__main__':
    main()