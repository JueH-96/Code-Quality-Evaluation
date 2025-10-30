# YOUR CODE HERE
n, m = map(int, input().split())
products = []
for i in range(n):
    line = list(map(int, input().split()))
    price = line[0]
    count = line[1]
    functions = set(line[2:])
    products.append((price, functions))

ans = 'No'
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        price_i, functions_i = products[i]
        price_j, functions_j = products[j]
        if price_i >= price_j and functions_i.issubset(functions_j):
            if price_i > price_j or not functions_j.issubset(functions_i):
                ans = 'Yes'
                break
    if ans == 'Yes':
        break

print(ans)