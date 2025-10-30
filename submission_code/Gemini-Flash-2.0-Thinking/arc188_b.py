import math

def solve():
    n, k = map(int, input().split())

    if math.gcd(n, k) == 1:
        print("Yes")
    elif n % 2 == 0 and k == n // 2:
        print("No")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()