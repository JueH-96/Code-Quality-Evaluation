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
    
    max_product = 0
    for i in range(n):
        new_product = 1
        for j in range(n):
            if i == j:
                new_product *= (a[j] + 1)
            else:
                new_product *= a[j]
        if new_product > max_product:
            max_product = new_product
    
    results.append(max_product)

for result in results:
    print(result)