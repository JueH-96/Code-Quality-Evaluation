# YOUR CODE HERE
import sys

def max_diff(n, a):
    max_diff = 0
    for k in range(1, n + 1):
        if n % k == 0:
            truck_weights = [sum(a[i * k:(i + 1) * k]) for i in range(n // k)]
            max_diff = max(max_diff, max(truck_weights) - min(truck_weights))
    return max_diff

input = sys.stdin.read
data = input().split()
t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n
    results.append(max_diff(n, a))

for result in results:
    print(result)