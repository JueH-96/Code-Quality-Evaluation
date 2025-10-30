N, M = map(int, input().split())
A = list(map(int, input().split()))

count = 0
for s in range(N):
    dist = 0
    for k in range(1, N):
        dist += A[(s + k - 1) % N]
        if dist % M == 0:
            count += 1

print(count)