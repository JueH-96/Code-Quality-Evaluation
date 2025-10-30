N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

total_price = 0
for color in C:
    price = P[0]
    for i in range(M):
        if color == D[i]:
            price = P[i+1]
            break
    total_price += price

print(total_price)