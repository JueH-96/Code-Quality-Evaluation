# YOUR CODE HERE
N, M = map(int, input().split())

products = []
for _ in range(N):
    line = list(map(int, input().split()))
    price = line[0]
    count = line[1]
    functions = set(line[2:2+count])
    products.append((price, functions))

# Check all pairs
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        
        price_i, func_i = products[i]
        price_j, func_j = products[j]
        
        # Condition 1: P_i >= P_j
        if price_i >= price_j:
            # Condition 2: j has all functions of i
            if func_i.issubset(func_j):
                # Condition 3: P_i > P_j OR j has a function that i doesn't have
                if price_i > price_j or len(func_j - func_i) > 0:
                    print("Yes")
                    exit()

print("No")