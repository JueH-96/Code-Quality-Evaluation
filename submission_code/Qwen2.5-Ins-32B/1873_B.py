import sys

def max_product_after_increment(t, test_cases):
    results = []
    for n, digits in test_cases:
        max_product = 0
        for i in range(n):
            new_digits = digits[:]
            new_digits[i] += 1
            product = 1
            for digit in new_digits:
                product *= digit
            max_product = max(max_product, product)
        results.append(max_product)
    return results

t = int(input())
test_cases = []
for _ in range(t):
    n = int(input())
    digits = list(map(int, input().split()))
    test_cases.append((n, digits))

results = max_product_after_increment(t, test_cases)
for result in results:
    print(result)