N, K = map(int, input().split())
half = (N*K+1)//2
for i in range(1, N+1):
    for _ in range(min(K, max(0, half-(i-1)*K))):
        print(i, end=' ')
    half -= min(K, max(0, half-(i-1)*K))
for i in range(N, 0, -1):
    for _ in range(K, min(K, max(0, half-(i-1)*K)), -1):
        print(i, end=' ')
    half -= min(K, max(0, half-(i-1)*K))
print()