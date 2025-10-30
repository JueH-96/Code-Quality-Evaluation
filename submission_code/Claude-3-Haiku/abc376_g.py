import sys

def solve_case():
    N = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))

    total_a = sum(a)
    expected_operations = 0
    for i in range(1, N+1):
        if p[i-1] == 0:
            continue
        probability = a[i-1] / total_a
        expected_operations += probability * (i - p[i-1])

    return int(expected_operations % 998244353)

T = int(input())
for _ in range(T):
    print(solve_case())