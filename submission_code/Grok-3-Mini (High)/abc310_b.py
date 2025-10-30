import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read N and M
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Store products as list of (price, function set)
products = []
for _ in range(N):
    P = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    func_set = set(map(int, data[index:index + C]))
    index += C
    products.append((P, func_set))

# Check for any pair (i, j) that satisfies the conditions
found = False
for i in range(N):
    P_i, func_i = products[i]
    for j in range(N):
        P_j, func_j = products[j]
        if P_i >= P_j and func_i.issubset(func_j) and (P_i > P_j or len(func_j) > len(func_i)):
            found = True
            break  # Break inner loop
    if found:
        break  # Break outer loop

# Output the result
if found:
    print("Yes")
else:
    print("No")