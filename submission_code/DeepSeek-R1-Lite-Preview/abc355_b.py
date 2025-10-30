# Read N and M
N, M = map(int, input().split())

# Read sequence A
A = list(map(int, input().split()))

# Read sequence B
B = list(map(int, input().split()))

# Convert A to a set for O(1) lookups
A_set = set(A)

# Combine A and B, and sort to get C
C = sorted(A + B)

# Iterate through C and check for consecutive elements from A
for i in range(len(C) - 1):
    if C[i] in A_set and C[i+1] in A_set:
        print("Yes")
        break
else:
    print("No")