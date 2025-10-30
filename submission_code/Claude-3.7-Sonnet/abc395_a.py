# Read number of elements
N = int(input())

# Read the sequence
A = list(map(int, input().split()))

# Check if the sequence is strictly increasing
is_strictly_increasing = True
for i in range(N - 1):
    if A[i] >= A[i + 1]:
        is_strictly_increasing = False
        break

# Output result
if is_strictly_increasing:
    print("Yes")
else:
    print("No")