def solve():
    n, k = map(int, input().split())
    
    if n % 2 != 0:
        print("Yes")
        return

    if (k * 2) % n == 0:
        print("No")
    else:
        print("Yes")

t = int(input())
for _ in range(t):
    solve()