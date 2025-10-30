import math

def expected_salary(N, A):
    total = 0
    for i in range(N):
        total += (i + 1) * A[i]
    
    mod = 998244353
    denominator = 0
    for i in range(1, N + 1):
        denominator += i
    
    numerator = total % mod
    denominator = pow(denominator, mod - 2, mod)
    
    return numerator * denominator % mod

if __name__ == "__main__":
    N = int(input())
    A = [int(x) for x in input().split()]
    print(int(expected_salary(N, A)))