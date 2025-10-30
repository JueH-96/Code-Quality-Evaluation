import sys
import math

def solve():
    """
    This function reads an integer N and calculates the number of positive integers
    not greater than N that have exactly 9 positive divisors.
    """
    N = int(sys.stdin.readline())

    # An integer has exactly 9 divisors if it is of the form p^8 or p^2 * q^2
    # for distinct primes p and q. These two sets of numbers are disjoint.

    # Case 1: Numbers of the form p^8
    # We count primes p such that p^8 <= N.
    # This implies p <= N^(1/8). Since N <= 4 * 10^12, p is at most ~37.6.
    count1 = 0
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        # Python's arbitrary-precision integers handle p**8 without overflow.
        if p ** 8 <= N:
            count1 += 1
        else:
            # Since primes are sorted, no larger prime will satisfy the condition.
            break

    # Case 2: Numbers of the form p^2 * q^2, where p < q are primes.
    # This is equivalent to counting pairs (p, q) with p < q such that p*q <= sqrt(N).
    M = int(math.sqrt(N))
    count2 = 0

    # The smallest product of two distinct primes is 2*3=6. If M < 6, no such pairs exist.
    if M >= 6:
        # Generate primes up to M using a Sieve of Eratosthenes.
        is_prime = [True] * (M + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(M)) + 1):
            if is_prime[i]:
                for multiple in range(i * i, M + 1, i):
                    is_prime[multiple] = False
        
        primes = [i for i, is_p in enumerate(is_prime) if is_p]

        # Use a two-pointer approach to count valid pairs (p, q).
        num_primes = len(primes)
        if num_primes >= 2:
            left, right = 0, num_primes - 1
            while left < right:
                p = primes[left]
                q = primes[right]

                # A safe way to check if p * q > M, avoiding potential overflow in other languages.
                if p > M // q:
                    right -= 1
                else:
                    # If (p, q) is a valid pair, then for the fixed p (primes[left]),
                    # all primes from primes[left+1] to primes[right] will also
                    # form valid pairs. The number of such pairs is (right - left).
                    count2 += (right - left)
                    left += 1

    print(count1 + count2)

solve()