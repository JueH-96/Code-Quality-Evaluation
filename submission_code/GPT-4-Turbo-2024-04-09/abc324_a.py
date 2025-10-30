# Read the number of elements
N = int(input().strip())

# Read the elements
A = list(map(int, input().strip().split()))

# Check if all elements are the same
if all(x == A[0] for x in A):
    print("Yes")
else:
    print("No")