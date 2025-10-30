# Read input from stdin
N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort the combined list of A and B
C = sorted(A + B)

# Check if there are two consecutive elements from A in C
found_consecutive = False
for i in range(1, len(C)):
    if C[i] in A and C[i-1] in A:
        found_consecutive = True
        break

# Print the result
print("Yes" if found_consecutive else "No")