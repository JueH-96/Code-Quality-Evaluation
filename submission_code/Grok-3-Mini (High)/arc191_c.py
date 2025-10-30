import math
import sys

# Miller-Rabin primality test
def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False
    s = 0
    d = n - 1
    while d % 2 == 0:
        d //= 2
        s += 1
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            break
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for r in range(1, s):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            if x != n - 1:
                return False
    return True

# Get all divisors of n, sorted
def get_divisors(n):
    divs = set()
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(list(divs))

# Read input
data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1

for test in range(T):
    N = int(data[index])
    index += 1
    if N == 1:
        # Special case for N=1
        print(1, 1)
        continue
    # Find a prime M such that M ≡ 1 mod N
    k = 1
    M_found = False
    M = 0
    while not M_found:
        M = k * N + 1
        if M > 2e18:  # Safety check to prevent infinite loop, though unlikely
            sys.exit("Error: no prime found for M")  # Should not happen
        if is_prime(M):
            M_found = True
        else:
            k += 1
    # M is now a prime with M ≡ 1 mod N
    # Get divisors of N once
    divs = get_divisors(N)
    # Find A such that the order is exactly N
    A_found = False
    for A_try in range(2, 1002):  # Try A from 2 to 1001
        if pow(A_try, N, M) == 1:  # A^N ≡ 1 mod M
            order_is_N = True
            for d in divs:
                if d < N and pow(A_try, d, M) == 1:
                    order_is_N = False
                    break
            if order_is_N:
                print(A_try, M)
                A_found = True
                break  # Found A, move to next test case
    if not A_found:
        # If no A found in range, this should not happen, but handle it
        sys.exit("Error: no A found for order N")  # Should not happen with the range