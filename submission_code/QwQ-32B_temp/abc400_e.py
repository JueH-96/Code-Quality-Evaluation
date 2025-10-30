import sys
import math

def main():
    max_n = 10**6
    # Precompute omega array
    omega = [0] * (max_n + 1)
    for p in range(2, max_n + 1):
        if omega[p] == 0:  # p is a prime
            for multiple in range(p, max_n + 1, p):
                omega[multiple] += 1

    # Precompute max_two_primes array
    max_two_primes = [0] * (max_n + 1)
    current_max = 0
    for x in range(1, max_n + 1):
        if omega[x] == 2:
            current_max = x
        else:
            current_max = max_two_primes[x-1]
        max_two_primes[x] = current_max

    # Read input
    input = sys.stdin.read().split()
    Q = int(input[0])
    queries = map(int, input[1:Q+1])

    # Process each query
    for A in queries:
        s_max = int(math.isqrt(A))
        s = max_two_primes[s_max]
        print(s * s)

if __name__ == "__main__":
    main()