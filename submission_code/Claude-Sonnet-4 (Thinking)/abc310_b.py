n, m = map(int, input().split())
products = []

for _ in range(n):
    line = list(map(int, input().split()))
    price = line[0]
    c = line[1]
    functions = set(line[2:])
    products.append((price, functions))

found = False
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        
        price_i, functions_i = products[i]
        price_j, functions_j = products[j]
        
        # Check condition 1: P_i >= P_j
        if price_i < price_j:
            continue
        
        # Check condition 2: j has all functions of i
        if not functions_i.issubset(functions_j):
            continue
        
        # Check condition 3: P_i > P_j OR j has strictly more functions than i
        if price_i > price_j or len(functions_j) > len(functions_i):
            found = True
            break
    
    if found:
        break

print("Yes" if found else "No")