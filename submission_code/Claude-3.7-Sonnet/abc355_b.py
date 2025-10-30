# Read inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Create a set to quickly determine if an element is from sequence A
A_set = set(A)

# Form sequence C by merging and sorting
C = sorted(A + B)

# Check for consecutive elements from A
for i in range(1, len(C)):
    if C[i] in A_set and C[i-1] in A_set:
        print("Yes")
        break
else:
    print("No")