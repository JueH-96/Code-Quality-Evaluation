# Read input
N = int(input())
A = list(map(int, input().split()))

# Check for three consecutive equal elements
if any(A[i] == A[i+1] == A[i+2] for i in range(len(A) - 2)):
    print("Yes")
else:
    print("No")