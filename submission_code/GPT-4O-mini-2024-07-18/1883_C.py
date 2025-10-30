def min_operations_to_make_product_divisible(t, test_cases):
    results = []
    
    for n, k, a in test_cases:
        # Count the number of operations needed
        operations_needed = 0
        
        # Check the remainder of each number when divided by k
        remainder_count = [0] * k
        
        for number in a:
            remainder_count[number % k] += 1
        
        # We need to ensure that the product is divisible by k
        if k == 2:
            # We need at least one even number
            if remainder_count[0] == 0:
                # No even number, we need to increment one odd number
                operations_needed = 1
        elif k == 3:
            # We need at least one number that is 0 mod 3 or two numbers that are 1 mod 3
            if remainder_count[0] == 0 and remainder_count[1] < 2:
                operations_needed = 2 - remainder_count[1]
        elif k == 4:
            # We need at least one number that is 0 mod 4 or two numbers that are 2 mod 4
            if remainder_count[0] == 0 and remainder_count[2] < 2:
                operations_needed = 2 - remainder_count[2]
        elif k == 5:
            # We need at least one number that is 0 mod 5
            if remainder_count[0] == 0:
                operations_needed = 1
        
        results.append(operations_needed)
    
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

t = int(data[0])
test_cases = []

index = 1
for _ in range(t):
    n, k = map(int, data[index].split())
    a = list(map(int, data[index + 1].split()))
    test_cases.append((n, k, a))
    index += 2

# Get results
results = min_operations_to_make_product_divisible(t, test_cases)

# Print results
for result in results:
    print(result)