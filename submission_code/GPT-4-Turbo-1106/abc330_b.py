def find_closest_values(N, L, R, A):
    X = []
    for a in A:
        if a < L:
            X.append(L)
        elif a > R:
            X.append(R)
        else:
            X.append(a)
    return X

# Read input from stdin
N, L, R = map(int, input().split())
A = list(map(int, input().split()))

# Find the closest values
X = find_closest_values(N, L, R, A)

# Write output to stdout
print(' '.join(map(str, X)))