n = int(input())
buildings = []
for _ in range(n):
    x, h = map(int, input().split())
    buildings.append((x, h))

max_threshold = float('-inf')

for i in range(n):
    xi, hi = buildings[i]
    threshold_i = float('-inf')
    
    for j in range(i):
        xj, hj = buildings[j]
        threshold = (hj * xi - hi * xj) / (xi - xj)
        threshold_i = max(threshold_i, threshold)
    
    max_threshold = max(max_threshold, threshold_i)

if max_threshold < 0:
    print(-1)
else:
    print(max_threshold)