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
    
    count = {}
    for ai in a:
        if ai in count:
            count[ai] += 1
        else:
            count[ai] = 1
    
    result = 0
    for ai in count:
        if count[ai] > 1:
            result += count[ai] * (count[ai] - 1) // 2
    
    results.append(result)

for result in results:
    print(result)