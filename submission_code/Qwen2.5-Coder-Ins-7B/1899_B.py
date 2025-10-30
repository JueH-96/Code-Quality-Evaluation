# YOUR CODE HERE
import sys
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
    
    a.sort()
    max_diff = 0
    
    for k in range(1, n + 1):
        if n % k == 0:
            total_weight = sum(a[i] for i in range(n) if (i // k) % 2 == 0)
            max_diff = max(max_diff, total_weight - (sum(a) - total_weight))
    
    results.append(max_diff)

for result in results:
    print(result)