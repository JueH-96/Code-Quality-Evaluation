from typing import List, Tuple

def max_prints(N: int, products: List[Tuple[int, int]]) -> int:
    # Sort the products by their exit time
    products.sort(key=lambda x: x[1])
    
    # Initialize the number of prints and the current time
    num_prints = 0
    current_time = 0
    
    for t, d in products:
        # If the product can be printed on, do so
        if t >= current_time:
            num_prints += 1
            current_time = t + d + 1
    
    return num_prints

# Read the input
N = int(input())
products = []
for _ in range(N):
    t, d = map(int, input().split())
    products.append((t, d))

# Solve the problem and print the answer
print(max_prints(N, products))