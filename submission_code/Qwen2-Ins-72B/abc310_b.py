N, M = map(int, input().split())
products = []
for _ in range(N):
    P, C, *F = map(int, input().split())
    products.append((P, set(F)))

def is_strictly_superior(i, j):
    pi, fi = products[i]
    pj, fj = products[j]
    if pi >= pj and fi.issubset(fj):
        if pi > pj or len(fj - fi) > 0:
            return True
    return False

for i in range(N):
    for j in range(N):
        if i != j and is_strictly_superior(i, j):
            print("Yes")
            exit()
print("No")