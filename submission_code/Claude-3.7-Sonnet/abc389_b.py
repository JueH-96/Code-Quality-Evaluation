# YOUR CODE HERE
X = int(input())
N = 1
current_factorial = 1

while current_factorial < X:
    N += 1
    current_factorial *= N

print(N)