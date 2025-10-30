def solve():
    N, M = map(int, input().split())
    products = []
    
    for i in range(N):
        data = list(map(int, input().split()))
        price = data[0]
        num_functions = data[1]
        functions = set(data[2:])
        products.append((price, functions))
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            
            price_i, functions_i = products[i]
            price_j, functions_j = products[j]
            
            # Condition 1: P_i >= P_j
            if price_i < price_j:
                continue
            
            # Condition 2: j has all functions of i
            if not functions_i.issubset(functions_j):
                continue
            
            # Condition 3: P_i > P_j or j has a function that i doesn't have
            if price_i > price_j or functions_i != functions_j:
                return "Yes"
    
    return "No"

print(solve())