# YOUR CODE HERE
X = int(input())

n = 1
factorial = 1

while factorial < X:
    n += 1
    factorial *= n

print(n)