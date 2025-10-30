# YOUR CODE HERE
N, M = map(int, input().split())
products = []

for _ in range(N):
    line = list(map(int, input().split()))
    price, num_functions = line[:2]
    functions = set(line[2:])
    products.append((price, functions))

for i in range(N):
    for j in range(N):
        if i != j:
            price_i, functions_i = products[i]
            price_j, functions_j = products[j]
            
            if price_i >= price_j and functions_i.issubset(functions_j):
                if price_i > price_j or len(functions_j) > len(functions_i):
                    print("Yes")
                    exit()

print("No")