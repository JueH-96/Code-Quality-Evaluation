# YOUR CODE HERE
# Read N
N = int(input())

# Read the array
A = list(map(int, input().split()))

# Check for three consecutive equal elements
found = False
for i in range(N - 2):
    if A[i] == A[i + 1] == A[i + 2]:
        found = True
        break

# Output result
if found:
    print("Yes")
else:
    print("No")