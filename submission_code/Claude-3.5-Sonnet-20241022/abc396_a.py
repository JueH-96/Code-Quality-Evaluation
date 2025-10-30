# Read input
N = int(input())
A = list(map(int, input().split()))

# Check for three consecutive equal elements
has_three_consecutive = False
for i in range(N-2):
    if A[i] == A[i+1] == A[i+2]:
        has_three_consecutive = True
        break

# Output result
print("Yes" if has_three_consecutive else "No")