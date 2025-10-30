import sys

def solve():
    N = int(sys.stdin.readline().strip())
    times = list(map(int, sys.stdin.readline().strip().split()))
    delays = list(map(int, sys.stdin.readline().strip().split()))
    max_products = 0
    for i in range(N):
        if times[i] <= max_products:
            max_products += delays[i]
        else:
            max_products += times[i]
    print(max_products)

solve()