# YOUR CODE HERE
X = int(input())
total = 0
for i in range(1, 10):
    for j in range(1, 10):
        if i * j != X:
            total += i * j
print(total)