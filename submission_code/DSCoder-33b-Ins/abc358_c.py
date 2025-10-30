# YOUR CODE HERE
N, M = map(int, input().split())
S = [input() for _ in range(N)]

def solve(N, M, S):
    ans = M
    for i in range(1, 2**N):
        buy = [False]*M
        cnt = bin(i).count('1')
        for j in range(N):
            if ((i >> j) & 1):
                for k in range(M):
                    buy[k] = buy[k] or (S[j][k] == 'o')
        if all(buy):
            ans = min(ans, cnt)
    return ans

print(solve(N, M, S))