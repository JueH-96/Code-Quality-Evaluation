# Read the input
N = int(input())
A = list(map(int, input().split()))

# Check for three consecutive equal elements
found = False
for i in range(N - 2):
    if A[i] == A[i+1] == A[i+2]:
        found = True
        break

# Output the result
if found:
    print("Yes")
else:
    print("No")