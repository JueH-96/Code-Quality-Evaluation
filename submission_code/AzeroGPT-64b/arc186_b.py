def main():
    from sys import stdin
    input = stdin.readline

    MOD = 998244353

    N = int(input())
    A = list(map(int, input().split()))

    # 累積和
    acc = [1] * N
    
    # 各入力に対する値
    v = [0] * N

    for i, a in enumerate(A):
        if a > 0:
            v[i] = v[i - (a + 1)] + 1
    
    # 最後の要素はNから値を取り除く
    for i in range(N - 1, -1, -1):
        acc[i] = (acc[i + 1] + v[i]) % MOD
        v[i] -= 1

    # 二項係数のテーブル
    idx = bit_length(N - 1)
    table = [[1] * (N + 1) for _ in range(idx)]
    for i in range(1, idx):
        for j in range(N + 1):
            table[i][j] = table[i - 1][j]
            if j:
                table[i][j] += table[i - 1][j - 1]
            table[i][j] %= MOD

    tol = [1] * N

    for i in range(N):
        n = N - i - 1 - v[i]
        r = tol[i] * pow(2, n, MOD) % MOD
        r = table[idx - 1][n] * r % MOD
        if r > 0:
            tol[i + 1] = r
        else:
            tol[i + 1] = tol[i] * pow(2, N - i - 1 - v[i], MOD) % MOD
        tol[i + 1] *= acc[i + 1]
        tol[i + 1] %= MOD

    print(tol[N])

def bit_length(x):
    x += 1
    l = 0
    while x > 0:
        x >>= 1
        l += 1
    return l

if __name__ == '__main__':
    main()