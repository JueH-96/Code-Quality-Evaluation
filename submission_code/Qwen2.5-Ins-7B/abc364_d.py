# N, Q = map(int, input().split())
# points_a = list(map(int, input().split()))
# queries = [list(map(int, input().split())) for _ in range(Q)]

N, Q = 10, 5
points_a = [-84, -60, -41, -100, 8, -8, -52, -62, -61, -76]
queries = [[-52, 5], [14, 4], [-2, 6], [46, 2], [26, 7]]

points_a.sort()
for b, k in queries:
    distances = sorted(abs(a - b) for a in points_a)
    print(distances[k - 1])