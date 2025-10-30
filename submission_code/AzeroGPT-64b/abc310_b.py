N, M = map(int, input().split())
products = []

for _ in range(N):
    P, C, *functions = map(int, input().split())
    products.append((P, set(functions)))

# Check if there is a product that is strictly superior to another
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        if products[i][0] >= products[j][0] and products[j][1].issuperset(products[i][1]):
            if products[i][0] > products[j][0] or len(products[j][1] - products[i][1]) > 0:
                print("Yes")
                return

print("No")