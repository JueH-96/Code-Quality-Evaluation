# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

result = [0] * N
next_firework = 0

for i in range(N - 1, -1, -1):
    if next_firework < M and A[next_firework] - 1 == i:
        result[i] = 0
        next_firework += 1
    elif i == N - 1:
        result[i] = 0
    else:
        result[i] = result[i + 1] + 1

for r in result:
    print(r)