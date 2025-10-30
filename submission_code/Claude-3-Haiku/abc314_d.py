# Read the input
N = int(input())
S = input()
Q = int(input())

# Perform the operations
for _ in range(Q):
    t, x, c = map(int, input().split())
    if t == 1:
        S = S[:x-1] + c + S[x:]
    elif t == 2:
        S = S.lower()
    else:
        S = S.upper()

# Print the final string
print(S)