# YOUR CODE HERE
N, Q = map(int, input().split())
a = list(map(int, input().split()))

for _ in range(Q):
    b, k = map(int, input().split())
    distances = []
    for ai in a:
        distances.append(abs(b - ai))
    distances.sort()
    print(distances[k-1])