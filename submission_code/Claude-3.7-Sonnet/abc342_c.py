# Read input
N = int(input())
S = input()
Q = int(input())

# Perform Q operations
for _ in range(Q):
    c_i, d_i = input().split()
    S = S.replace(c_i, d_i)

# Print the final string
print(S)