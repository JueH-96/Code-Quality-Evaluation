def main():
    N, M = map(int, input().split())
    A = [tuple(map(int, input().split())) for _ in range(N)]
    sum_f = 0
    dp = [0] * (M + 1)
    dp[0] = 1
    mod = 998244353

    for i in range(N):
        cnt = [0] * (M + 2)
        A_transformed = []
        for j in range(M):
            A_transformed.append((dp[j] + A[i][j]) % 2)
            cnt[A_transformed[-1]] += 1
            cnt[dp[j]] -= 1
            dp[j + 1] = A_transformed[-1]
        cnt[A_transformed[-1]] += 1
        dp = A_transformed + [0]

        for j in range(i, N):
            if j != i and A[i] != A[j][:M]:
                x = 0
                while A[i] != A[j][:M]:
                    A[j] = [(A_j_prev + A_j) % 2 for A_j_prev, A_j in zip(A[j], A[j][1:] + (0,))]
                    x += 1
                sum_f += x
            sum_f += cnt[A[j][0]]

    print(sum_f % mod)

if __name__ == '__main__':
    main()