def max_product_with_increment(t, test_cases):
    results = []
    for case in test_cases:
        n, digits = case
        max_product = 0
        
        for i in range(n):
            # Create a new list with one digit incremented by 1
            new_digits = digits[:]
            new_digits[i] += 1
            
            # Calculate the product of the new list
            product = 1
            for digit in new_digits:
                product *= digit
            
            # Update the maximum product found
            max_product = max(max_product, product)
        
        results.append(max_product)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []

index = 1
for _ in range(t):
    n = int(data[index])
    digits = list(map(int, data[index + 1].split()))
    test_cases.append((n, digits))
    index += 2

# Get results
results = max_product_with_increment(t, test_cases)

# Print results
for result in results:
    print(result)