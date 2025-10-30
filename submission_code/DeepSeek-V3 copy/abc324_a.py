# Read the input values
N = int(input())
A = list(map(int, input().split()))

# Check if all elements are equal
if all(a == A[0] for a in A):
    print("Yes")
else:
    print("No")