# YOUR CODE HERE
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))

price_dict = {D[i]: P[i + 1] for i in range(M)}
total_price = sum(price_dict.get(c, P[0]) for c in C)

print(total_price)