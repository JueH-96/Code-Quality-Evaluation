import sys
import math

def count_divisors(n):
    count = 0
    sqrt_n = int(math.isqrt(n))
    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
    return count

def main():
    N = int(sys.stdin.read().strip())
    count = 0
    for i in range(1, N + 1):
        if count_divisors(i) == 9:
            count += 1
    sys.stdout.write(str(count) + '
')

if __name__ == "__main__":
    main()