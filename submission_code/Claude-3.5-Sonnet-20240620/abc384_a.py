# YOUR CODE HERE
# Read input
N, c1, c2 = input().split()
N = int(N)
S = input().strip()

# Replace characters
result = ''.join([c if c == c1 else c2 for c in S])

# Print output
print(result)