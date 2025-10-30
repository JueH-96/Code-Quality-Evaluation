# YOUR CODE HERE
N, M = map(int, input().split())

products = []
for _ in range(N):
    line = list(map(int, input().split()))
    price = line[0]
    count = line[1]
    functions = set(line[2:2+count])
    products.append((price, functions))

# Check all pairs (i, j)
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        
        price_i, funcs_i = products[i]
        price_j, funcs_j = products[j]
        
        # Condition 1: P_i >= P_j
        if price_i < price_j:
            continue
        
        # Condition 2: j has all functions of i
        if not funcs_i.issubset(funcs_j):
            continue
        
        # Condition 3: P_i > P_j OR j has extra functions
        if price_i > price_j or len(funcs_j) > len(funcs_i):
            print("Yes")
            exit()

print("No")