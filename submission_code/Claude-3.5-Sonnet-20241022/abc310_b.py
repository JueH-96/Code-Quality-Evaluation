N, M = map(int, input().split())
products = []

for _ in range(N):
    line = list(map(int, input().split()))
    price = line[0]
    num_funcs = line[1]
    funcs = set(line[2:])
    products.append((price, funcs))

found = False
for i in range(N):
    for j in range(N):
        if i == j:
            continue
            
        price_i, funcs_i = products[i]
        price_j, funcs_j = products[j]
        
        # Check if price of i is >= price of j
        if price_i >= price_j:
            # Check if j has all functions of i
            if funcs_i.issubset(funcs_j):
                # Check if price is strictly greater OR j has extra functions
                if price_i > price_j or len(funcs_j - funcs_i) > 0:
                    found = True
                    break
    if found:
        break

print("Yes" if found else "No")