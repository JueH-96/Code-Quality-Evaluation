def is_strictly_superior(products):
    for i in range(len(products)):
        for j in range(len(products)):
            if i != j:
                pi, ci, fi = products[i]
                pj, cj, fj = products[j]
                if pi >= pj and all(f in fj for f in fi) and (pi > pj or any(f not in fi for f in fj)):
                    return True
    return False

# Read input
N, M = map(int, input().split())
products = []
for _ in range(N):
    data = list(map(int, input().split()))
    P = data[0]
    C = data[1]
    F = data[2:]
    products.append((P, C, F))

# Check if any product is strictly superior
answer = "Yes" if is_strictly_superior(products) else "No"

# Output the answer
print(answer)