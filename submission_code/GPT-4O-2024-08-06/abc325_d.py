# YOUR CODE HERE
def max_products_to_print(n, products):
    # Sort products by their end time (T_i + D_i)
    products.sort(key=lambda x: x[0] + x[1])
    
    current_time = 0
    count = 0
    
    for T, D in products:
        if current_time <= T + D:
            # We can print on this product
            count += 1
            # Update current time to the end time of this product plus recharge time
            current_time = T + D + 1
    
    return count

import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
products = []

index = 1
for _ in range(n):
    T = int(data[index])
    D = int(data[index + 1])
    products.append((T, D))
    index += 2

result = max_products_to_print(n, products)
print(result)