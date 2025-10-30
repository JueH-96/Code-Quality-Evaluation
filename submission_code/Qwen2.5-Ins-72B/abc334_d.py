# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
R = list(map(int, data[2:N+2]))
queries = list(map(int, data[N+2:]))

R.sort()

prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + R[i]

def max_sleighs(x):
    low, high = 0, N
    while low < high:
        mid = (low + high + 1) // 2
        if prefix_sum[mid] <= x:
            low = mid
        else:
            high = mid - 1
    return low

results = []
for x in queries:
    results.append(max_sleighs(x))

for result in results:
    print(result)