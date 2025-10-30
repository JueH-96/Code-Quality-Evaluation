import sys

def solve(n):
    total_cost = 0
    while n >= 2:
        if n % 2 == 0:
            n = n // 2
            total_cost += n
        else:
            n = (n + 1) // 2
            total_cost += n
    return total_cost

n = int(sys.stdin.readline().strip())
print(solve(n))