# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
result = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n

    max_product = 0

    for i in range(n):
        if a[i] != 9:
            a[i] += 1
            product = 1
            for num in a:
                product *= num
            max_product = max(max_product, product)
            a[i] -= 1

    result.append(max_product)

for res in result:
    print(res)