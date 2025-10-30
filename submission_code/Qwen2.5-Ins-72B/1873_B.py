# YOUR CODE HERE
import sys

def max_product_after_adding_one(n, a):
    if 0 in a:
        # If there's a zero, the best move is to add 1 to it
        a[a.index(0)] += 1
    else:
        # Otherwise, add 1 to the smallest digit
        min_index = a.index(min(a))
        a[min_index] += 1
    
    # Calculate the product of the array
    product = 1
    for digit in a:
        product *= digit
    
    return product

# Read the number of test cases
t = int(input().strip())

# Process each test case
for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().strip().split()))
    print(max_product_after_adding_one(n, a))