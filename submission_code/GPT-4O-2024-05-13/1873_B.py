# YOUR CODE HERE
def max_product_after_increment(t, test_cases):
    results = []
    for i in range(t):
        n = test_cases[i][0]
        digits = test_cases[i][1]
        
        max_product = 0
        for j in range(n):
            new_digits = digits[:]
            new_digits[j] += 1
            product = 1
            for digit in new_digits:
                product *= digit
            max_product = max(max_product, product)
        
        results.append(max_product)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
test_cases = []
for _ in range(t):
    n = int(data[index])
    digits = list(map(int, data[index + 1: index + 1 + n]))
    test_cases.append((n, digits))
    index += 1 + n

results = max_product_after_increment(t, test_cases)
for result in results:
    print(result)