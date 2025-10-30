N, Q = map(int, input().split())
a = list(map(int, input().split()))

for _ in range(Q):
    b, k = map(int, input().split())
    distances = [abs(a[i] - b) for i in range(N)]
    distances.sort()
    print(distances[k-1])  # k-1 because of 0-indexing