def find_lexicographically_smallest_path(S, T):
    X = []
    for i in range(len(S)):
        if S[i] != T[i]:
            S = S[:i] + T[i] + S[i+1:]
            X.append(S)
    return X

# Read input
S = input().strip()
T = input().strip()

# Find the lexicographically smallest path
result = find_lexicographically_smallest_path(S, T)

# Output the result
print(len(result))
for s in result:
    print(s)