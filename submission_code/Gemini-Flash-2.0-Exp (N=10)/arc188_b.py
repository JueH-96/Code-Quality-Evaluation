def solve():
    t = int(input())
    for _ in range(t):
        n, k = map(int, input().split())
        if n % 2 == 0 and k == n // 2:
            print("No")
        elif n % 2 == 0 and k % 2 == 0:
            print("Yes")
        elif n % 2 == 1:
            print("Yes")
        else:
            print("Yes")

solve()