from typing import List
from bisect import bisect_right

INF = 10 ** 18

# 解説AC
def main():
    N = int(input())
    D = list(map(int, input().split()))

    L1, C1, K1 = map(int, input().split())
    L2, C2, K2 = map(int, input().split())

    dp = [INF] * 100001
    dp[0] = 0

    for l, c, k in ((L1, C1, K1), (L2, C2, K2)):
        dp_k = [INF] * 100001
        for i in range(100001):
            if dp[i] == INF:
                continue
            dp_k[i] = dp[i]
            for j in range(1, k + 1):
                dp_k[i + j * l] = min(dp_k[i + j * l], dp[i] + j * c)
        dp = dp_k

    ans = sum(dp[d] for d in D)
    if ans >= INF:
        print(-1)
    else:
        print(ans)

if __name__ == '__main__':
    main()