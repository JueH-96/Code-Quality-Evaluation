# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    s = data[index + 2]
    index += 3
    
    count = 0
    i = 0
    while i < n:
        if s[i] == 'B':
            count += 1
            i += k
        else:
            i += 1
    
    results.append(count)

for result in results:
    print(result)