# Read the input values
N = int(input().strip())
S = input().strip()
Q = int(input().strip())

# Perform the operations
for _ in range(Q):
    c, d = input().strip().split()
    S = S.replace(c, d)

# Print the final string
print(S)