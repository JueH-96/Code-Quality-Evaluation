def is_strictly_superior(N, M, products):
    for i in range(N):
        price_i, functions_i = products[i]
        set_functions_i = set(functions_i)
        
        for j in range(N):
            if i == j:
                continue
            
            price_j, functions_j = products[j]
            set_functions_j = set(functions_j)
            
            if price_i >= price_j and set_functions_i.issubset(set_functions_j):
                if price_i > price_j or not set_functions_i.issubset(set_functions_j):
                    return "Yes"
    
    return "No"

import sys
input = sys.stdin.read
data = input().splitlines()

N, M = map(int, data[0].split())
products = []

for line in data[1:N+1]:
    parts = list(map(int, line.split()))
    price = parts[0]
    count = parts[1]
    functions = parts[2:2 + count]
    products.append((price, functions))

result = is_strictly_superior(N, M, products)
print(result)