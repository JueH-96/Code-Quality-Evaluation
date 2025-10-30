import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0

# Read N, M, D
N = int(data[index])
M = int(data[index + 1])
D = int(data[index + 2])
index += 3

# Read list A
A = list(map(int, data[index:index + N]))
index += N

# Read list B
B = list(map(int, data[index:index + M]))

# Sort both arrays in ascending order
A.sort()
B.sort()

# Two pointers starting from the end
i = N - 1
j = M - 1

# Flag to check if a valid pair is found
found = False
ans = 0

# Traverse using two pointers
while i >= 0 and j >= 0:
    diff = abs(A[i] - B[j])
    if diff <= D:
        # Valid pair found, calculate sum
        ans = A[i] + B[j]
        found = True
        break  # No need to continue as this is the maximum sum pair
    elif A[i] > B[j]:
        # A[i] is too large, move to a smaller A
        i -= 1
    else:
        # B[j] is too large, move to a smaller B
        j -= 1

# Output the result
if found:
    print(ans)
else:
    print(-1)