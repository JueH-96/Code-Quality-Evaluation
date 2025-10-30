# YOUR CODE HERE
n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create set of A elements for quick lookup
A_set = set(A)

# Merge and sort all elements
C = sorted(A + B)

# Check if any two consecutive elements in C are both from A
found = False
for i in range(len(C) - 1):
    if C[i] in A_set and C[i+1] in A_set:
        found = True
        break

if found:
    print("Yes")
else:
    print("No")