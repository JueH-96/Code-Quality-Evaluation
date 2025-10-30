# Read input
N = int(input())
A = list(map(int, input().split()))

# Check if strictly increasing
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