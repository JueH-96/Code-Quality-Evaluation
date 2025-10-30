# YOUR CODE HERE
n = int(input())
nearest = min(n % 5, 5 - n % 5)
print(n + nearest)