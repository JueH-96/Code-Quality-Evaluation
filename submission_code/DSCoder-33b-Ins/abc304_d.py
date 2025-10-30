# YOUR CODE HERE

# Read the inputs
W, H = map(int, input().split())
N = int(input())
strawberries = [tuple(map(int, input().split())) for _ in range(N)]
A = int(input())
a = list(map(int, input().split()))
B = int(input())
b = list(map(int, input().split()))

# Initialize the count of strawberries in each piece
counts = [[0]*(B+1) for _ in range(A+1)]

# Count the number of strawberries in each piece
for x, y in strawberries:
    i = j = 0
    while i < A and a[i] < x:
        i += 1
    while j < B and b[j] < y:
        j += 1
    counts[i][j] += 1

# Find the minimum and maximum number of strawberries
min_count = min(min(counts[i]) for i in range(A+1))
max_count = max(max(counts[i]) for i in range(A+1))

# Print the result
print(min_count, max_count)