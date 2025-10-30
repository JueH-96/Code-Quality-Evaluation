import sys
input = sys.stdin.read
N, S, Q, *CD = input().split()
S = list(S)
Q = int(Q)
CD = [CD[i:i+2] for i in range(0, len(CD), 2)]

# Create a mapping for all possible characters
mapping = {chr(i): chr(i) for i in range(ord('a'), ord('z') + 1)}

# Apply all operations to update the mapping
for c, d in CD:
    mapping[c] = d

# Apply the final mapping to the string S
for i in range(len(S)):
    S[i] = mapping[S[i]]

print(''.join(S))