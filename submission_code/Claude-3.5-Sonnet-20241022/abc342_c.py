# Read input
N = int(input())
S = list(input())
Q = int(input())

# Process each operation
for _ in range(Q):
    c, d = input().split()
    
    # Replace all occurrences of c with d
    for i in range(N):
        if S[i] == c:
            S[i] = d

# Print result
print(''.join(S))