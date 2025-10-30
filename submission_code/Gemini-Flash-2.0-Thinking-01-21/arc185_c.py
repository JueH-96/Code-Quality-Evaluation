def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if a[i] + a[j] + a[k] == x:
                    print(i + 1, j + 1, k + 1)
                    return
    print(-1)

if __name__ == '__main__':
    solve()