import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))

    total_salary = 0
    x = 0

    for a in A:
        total_salary += a
        x = max(x, a)

    expected_salary = total_salary / x
    expected_salary %= 998244353

    print(int(expected_salary))

solve()