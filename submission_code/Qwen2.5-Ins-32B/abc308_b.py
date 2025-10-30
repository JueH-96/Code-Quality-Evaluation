import sys

N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

price_dict = {D[i]: P[i+1] for i in range(M)}
price_dict['default'] = P[0]

total_price = sum(price_dict.get(color, price_dict['default']) for color in C)

print(total_price)