# YOUR CODE HERE
N, D = map(int, input().split())
snakes = []
for _ in range(N):
    T, L = map(int, input().split())
    snakes.append((T, L))

for k in range(1, D + 1):
    max_weight = max(T * (L + k) for T, L in snakes)
    print(max_weight)