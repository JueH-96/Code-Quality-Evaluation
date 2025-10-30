def max_product(test_cases):
    results = []
    for case in test_cases:
        n, digits = case
        max_prod = 0
        for i in range(n):
            temp = digits.copy()
            temp[i] += 1
            product = 1
            for num in temp:
                product *= num
            if product > max_prod:
                max_prod = product
        results.append(max_prod)
    return results

# Read input
t = int(input())
test_cases = []
for _ in range(t):
    n = int(input())
    digits = list(map(int, input().split()))
    test_cases.append((n, digits))

# Compute results
results = max_product(test_cases)

# Output results
for res in results:
    print(res)