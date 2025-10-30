# Read input
S = input().strip()
T = input().strip()

i = j = 0
result = []

# Process each character in S and T
while i < len(S) and j < len(T):
    if S[i] == T[j]:
        result.append(j + 1)
        i += 1
    j += 1

# Output the result
print(' '.join(map(str, result)))