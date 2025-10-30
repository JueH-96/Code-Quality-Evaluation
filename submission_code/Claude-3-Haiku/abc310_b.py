import sys

# Read input
N, M = map(int, input().split())
products = []
for _ in range(N):
    line = list(map(int, input().split()))
    price = line[0]
    num_functions = line[1]
    functions = sorted(line[2:])
    products.append((price, functions))

# Check if there is a strictly superior product
for i in range(N):
    for j in range(N):
        if i != j:
            if products[i][0] >= products[j][0] and all(f in products[j][1] for f in products[i][1]):
                if products[i][0] > products[j][0] or any(f not in products[i][1] for f in products[j][1]):
                    print("Yes")
                    sys.exit()

print("No")