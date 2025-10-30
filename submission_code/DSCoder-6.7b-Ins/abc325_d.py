import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    products = []
    for _ in range(N):
        T, D = map(int, sys.stdin.readline().strip().split())
        products.append((T, D))
    return N, products

def solve():
    N, products = read_input()
    products.sort(key=lambda x: (x[0] + x[1], x[0]))
    current_time = 0
    count = 0
    for T, D in products:
        if current_time + T <= D:
            count += 1
            current_time += T + 1
    return count

print(solve())