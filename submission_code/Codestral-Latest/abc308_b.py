# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

C = data[index:index + N]
index += N

D = data[index:index + M]
index += M

P = list(map(int, data[index:]))

price_dict = {D[i]: P[i + 1] for i in range(M)}
P_0 = P[0]

total_price = 0
for color in C:
    if color in price_dict:
        total_price += price_dict[color]
    else:
        total_price += P_0

print(total_price)