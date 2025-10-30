def solve():
    n, k = map(int, input().split())
    
    if n % 2 != 0:
        print("No")
        return

    if k % 2 != 0:
        print("Yes")
    else:
        print("No")


t = int(input())
for _ in range(t):
    solve()