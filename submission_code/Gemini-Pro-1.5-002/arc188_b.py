def solve():
    n, k = map(int, input().split())

    if n % 2 == 1:
        if k == n // 2:
            print("No")
        else:
            print("Yes")
    else:
        if k % 2 == 1:
            print("Yes")
        else:
            print("No")


t = int(input())
for _ in range(t):
    solve()