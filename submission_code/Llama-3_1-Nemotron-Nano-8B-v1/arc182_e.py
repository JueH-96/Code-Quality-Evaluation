import sys
import math

def main():
    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    A.sort()
    sum_A = sum(A)
    g = math.gcd(C, M)
    m_prime = M // g
    c_prime = C // g

    # Compute inv_c_prime modulo m_prime
    def modinv(a, m):
        g, x, y = extended_gcd(a, m)
        if g != 1:
            return None  # shouldn't happen as a and m are coprime
        else:
            return x % m

    def extended_gcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = extended_gcd(b % a, a)
            return (g, x - (b // a) * y, y)

    inv_c_prime = modinv(c_prime, m_prime)
    if inv_c_prime is None:
        inv_c_prime = 1

    # Compute sum_total
    sum_ms = 0
    for a in A:
        s_i = (M - a + g - 1) // g  # ceil((M -a)/g)
        sum_ms += (m_prime - s_i)
    sum_total = m_prime * sum_A - g * m_prime * sum_ms + (g * m_prime * (m_prime - 1) // 2) * N

    # Compute q and r_remain
    q = K // m_prime
    r_remain = K % m_prime

    # Compute sum_remain
    sum_remain = 0
    for k in range(r_remain):
        s = (k * c_prime) % m_prime
        x = g * s
        t = M - x
        if A[0] >= t:
            min_val = A[0] + x - M
        else:
            # Find the first A_i >= t using binary search
            left, right = 0, N
            while left < right:
                mid = (left + right) // 2
                if A[mid] < t:
                    left = mid + 1
                else:
                    right = mid
            if left < N:
                candidate2 = A[left] + x - M
                candidate1 = A[0] + x
                min_val = min(candidate1, candidate2)
            else:
                min_val = A[0] + x
        sum_remain += min_val

    total = q * sum_total + sum_remain
    print(total)

if __name__ == "__main__":
    main()