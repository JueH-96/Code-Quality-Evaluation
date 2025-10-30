# Read the input values
N = int(input())
A = []
for i in range(N):
    row = list(map(int, input().split()))
    A.append(row)

# Initialize the current element
current = 1

# Iterate through each element from 1 to N
for j in range(1, N+1):
    if current >= j:
        current = A[current-1][j-1]
    else:
        current = A[j-1][current-1]

# Print the final element
print(current)