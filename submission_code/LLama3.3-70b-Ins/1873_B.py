import sys

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    a.sort()
    a[0] += 1
    product = 1
    for num in a:
        product *= num
    print(product)

t = int(input())
for _ in range(t):
    solve()