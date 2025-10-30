# YOUR CODE HERE
N, M = map(int, input().split())
products = []
for _ in range(N):
    P, C = map(int, input().split())
    functions = set(map(int, input().split()))
    products.append((P, functions))

for i in range(N):
    for j in range(i + 1, N):
        if products[i][0] >= products[j][0]:
            if products[j][1].issuperset(products[i][1]):
                if products[i][0] > products[j][0] or len(products[j][1] - products[i][1]) > 0:
                    print('Yes')
                    exit()
print('No')