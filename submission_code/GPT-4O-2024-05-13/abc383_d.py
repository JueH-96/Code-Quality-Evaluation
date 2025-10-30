# YOUR CODE HERE
import sys
import math

def count_numbers_with_nine_divisors(N):
    def count_divisors(n):
        count = 0
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                if i * i == n:
                    count += 1
                else:
                    count += 2
        return count

    def is_prime(n):
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = []
    limit = int(N ** (1/3)) + 1
    for i in range(2, limit):
        if is_prime(i):
            primes.append(i)

    count = 0
    for p in primes:
        p3 = p ** 3
        if p3 <= N:
            count += 1
        else:
            break

    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            p1 = primes[i]
            p2 = primes[j]
            product = p1 * p1 * p2
            if product <= N:
                count += 1
            else:
                break

    return count

if __name__ == "__main__":
    input = sys.stdin.read().strip()
    N = int(input)
    result = count_numbers_with_nine_divisors(N)
    print(result)