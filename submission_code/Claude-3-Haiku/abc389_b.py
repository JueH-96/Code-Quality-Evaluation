import math

# Read the input from stdin
x = int(input())

# Find the positive integer N such that N! = X
n = 1
while True:
    if math.factorial(n) == x:
        print(n)
        break
    n += 1