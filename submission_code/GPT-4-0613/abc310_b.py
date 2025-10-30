N, M = map(int, input().split())
products = []
for _ in range(N):
    P, C, *F = map(int, input().split())
    products.append((P, set(F)))

for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if products[i][0] >= products[j][0] and products[i][1].issubset(products[j][1]) and (products[i][0] > products[j][0] or len(products[i][1]) < len(products[j][1])):
            print("Yes")
            exit(0)
print("No")