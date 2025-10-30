# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Take the last K elements and move them to the front
B = A[-K:] + A[:-K]

# Print the result
print(" ".join(map(str, B)))