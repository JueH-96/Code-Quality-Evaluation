def solve():
    n, k, x = map(int, input().split())
    a = list(map(int, input().split()))

    b = []
    for i in range(k):
        b.append(a[i])
    b.append(x)
    for i in range(k, n):
        b.append(a[i])

    print(*b)

if __name__ == "__main__":
    solve()