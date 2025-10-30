# YOUR CODE HERE
N = int(input())
products = []
for i in range(N):
    T, D = map(int, input().split())
    products.append((T, T + D))

products.sort()
printed = 0
last_printed_time = 0
i = 0
while i < N:
    best_product = -1
    best_end_time = -1
    for j in range(i, N):
        if products[j][0] >= last_printed_time + 1:
            if best_product == -1 or products[j][1] > best_end_time:
                best_product = j
                best_end_time = products[j][1]
    if best_product != -1:
        printed += 1
        last_printed_time = best_end_time
        i = best_product + 1
    else:
        break

print(printed)