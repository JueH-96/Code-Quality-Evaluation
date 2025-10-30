import sys
import bisect
import math

def main():
    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    d = math.gcd(C, M)
    m_prime = M // d
    sum_period = 0
    for k in range(m_prime):
        r = k * d
        b = M - r
        pos = bisect.bisect_left(A, b)
        candidate1 = r + A[0] if pos > 0 else float('inf')
        candidate2 = (A[pos] - b) if pos < N else float('inf')
        min_val = min(candidate1, candidate2)
        sum_period += min_val
    q, r = divmod(K, m_prime)
    sum_total = q * sum_period
    for k in range(r):
        rk = k * d
        b = M - rk
        pos = bisect.bisect_left(A, b)
        candidate1 = rk + A[0] if pos > 0 else float('inf')
        candidate2 = (A[pos] - b) if pos < N else float('inf')
        sum_total += min(candidate1, candidate2)
    print(sum_total)

if __name__ == '__main__':
    main()