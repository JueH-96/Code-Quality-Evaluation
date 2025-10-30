# Read the input
N = int(input())
S = input()
Q = int(input())
operations = []
for _ in range(Q):
    c, d = input().split()
    operations.append((c, d))

# Perform the operations
for c, d in operations:
    S = S.replace(c, d)

# Print the final string
print(S)