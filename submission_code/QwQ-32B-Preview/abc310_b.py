# Read N and M
N, M = map(int, input().split())

# Read N products
products = []
for _ in range(N):
    line = list(map(int, input().split()))
    P_i = line[0]
    C_i = line[1]
    functions_i = set(line[2:2+C_i])
    products.append((P_i, functions_i))

# Check for strictly superior pairs
found = False
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        P_i, functions_i = products[i]
        P_j, functions_j = products[j]
        if P_i >= P_j and functions_j <= functions_i and (P_i > P_j or functions_j < functions_i):
            found = True
            print("Yes")
            break
    if found:
        break
if not found:
    print("No")