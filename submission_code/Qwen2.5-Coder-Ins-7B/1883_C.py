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
    a = list(map(int, data[index + 2:index + 2 + n]))
    index += 2 + n
    
    count = 0
    for num in a:
        if num % k == 0:
            count += 1
    
    if count == 0:
        results.append(n)
    else:
        results.append(0)

for result in results:
    print(result)