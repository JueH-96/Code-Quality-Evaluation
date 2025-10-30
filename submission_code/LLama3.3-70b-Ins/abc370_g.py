import sys
import math

def get_divisors(n):
    """Get all divisors of a number."""
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

def is_good(n):
    """Check if a number is good."""
    return sum(get_divisors(n)) % 3 == 0

def count_sequences(N, M):
    """Count the number of sequences."""
    count = 0
    for i in range(1, N + 1):
        if is_good(i):
            count += 1
    return count ** M % 998244353

def main():
    """Main function."""
    N, M = map(int, sys.stdin.readline().split())
    print(count_sequences(N, M))

if __name__ == "__main__":
    main()