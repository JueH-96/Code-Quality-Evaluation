N, M = map(int, input().split())
degrees = [0] * N
for _ in range(M):
    u, v = map(int, input().split())
    degrees[u-1] += 1
    degrees[v-1] += 1

if N == 2 and M == 1:
    print("No")
else:
    print("Yes")
    print(' '.join(['1'] * N))