import sys
from functools import reduce
from operator import mul

def max_product_after_increment(digits):
    n = len(digits)
    max_product = 0

    for i in range(n):
        if digits[i] < 9:
            digits[i] += 1
            product = reduce(mul, digits, 1)
            digits[i] -= 1
            if product > max_product:
                max_product = product

    return max_product

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    t = int(data[index])
    index += 1
    results = []

    for _ in range(t):
        n = int(data[index])
        index += 1
        digits = list(map(int, data[index:index + n]))
        index += n
        results.append(max_product_after_increment(digits))

    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == "__main__":
    main()