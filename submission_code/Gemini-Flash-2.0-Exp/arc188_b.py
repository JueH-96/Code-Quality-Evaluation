def solve():
    n, k = map(int, input().split())
    if n % 2 == 0 and k == n // 2:
        print("No")
    else:
        print("Yes")

t = int(input())
for _ in range(t):
    solve()