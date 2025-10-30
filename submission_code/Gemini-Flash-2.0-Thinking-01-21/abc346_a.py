def solve():
    n = int(input())
    a = list(map(int, input().split()))

    b = []
    for i in range(n - 1):
        b_i = a[i] * a[i+1]
        b.append(b_i)

    print(*b)

solve()