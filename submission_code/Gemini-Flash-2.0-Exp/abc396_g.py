def solve():
    H, W = map(int, input().split())
    A = []
    for _ in range(H):
        A.append(input())

    ans = float('inf')

    for mask in range(2**W):
        B = []
        for i in range(H):
            row = ""
            for j in range(W):
                if (mask >> j) & 1:
                    row += str(1 - int(A[i][j]))
                else:
                    row += A[i][j]
            B.append(row)

        cur = 0
        for i in range(H):
            cnt0 = 0
            cnt1 = 0
            for j in range(W):
                if B[i][j] == '0':
                    cnt0 += 1
                else:
                    cnt1 += 1
            cur += min(cnt0, cnt1)
        ans = min(ans, cur)

    print(ans)

solve()