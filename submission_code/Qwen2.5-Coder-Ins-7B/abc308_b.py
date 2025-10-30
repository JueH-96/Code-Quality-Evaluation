# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
C = data[2:N+2]
D = data[N+2:N+2+M]
P = list(map(int, data[N+2+M:]))

price_dict = {D[i]: P[i+1] for i in range(M)}
price_dict[''] = P[0]

total_price = sum(price_dict[color] for color in C)
print(total_price)