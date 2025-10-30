import math

def gcd(a, b):
    """Compute the greatest common divisor of a and b"""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Compute the lowest common multiple of a and b"""
    return a * b // gcd(a, b)

def count_divisible_by_one_of(n, m, k):
    """Count the number of integers divisible by exactly one of n and m"""
    lcm_value = lcm(n, m)
    count = 0
    i = 1
    while True:
        if (i % n == 0 and i % m != 0) or (i % n != 0 and i % m == 0):
            count += 1
            if count == k:
                return i
        i += 1

def main():
    n, m, k = map(int, input().split())
    result = count_divisible_by_one_of(n, m, k)
    print(result)

if __name__ == "__main__":
    main()