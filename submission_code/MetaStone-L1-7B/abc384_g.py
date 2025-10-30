import bisect

n = int(input())
A = list(map(int, input().split()))
A.sort()

prefix_sum = [0] * (n + 1)
for i in range(n):
    prefix_sum[i + 1] = prefix_sum[i] + A[i]

k = int(input())
queries = []
for _ in range(k):
    x, y, B = list(map(int, input().split()))
    queries.append((x, y, B))

results = []
for x, y, B in queries:
    if y == 0:
        results.append(0)
        continue
    sum_B_Y = prefix_sum[y]
    sum_low_sum = 0
    for j in range(y):
        B_j_val = B[j]
        cnt = bisect.bisect_right(A, B_j_val, 0, x)
        sum_low_sum += prefix_sum[cnt]
    total = y * sum_B_Y - 2 * sum_low_sum
    results.append(total)

for result in results:
    print(result)