def transform_string(S, T):
    if S == T:
        return [0]
    
    n = len(S)
    X = []
    for i in range(n):
        if S[i] != T[i]:
            # Create a new string by changing the i-th character of S to T[i]
            new_S = S[:i] + T[i] + S[i+1:]
            X.append(new_S)
            S = new_S
            if S == T:
                break
    
    return [len(X)] + X

# Read input
S = input().strip()
T = input().strip()

# Solve the problem
result = transform_string(S, T)

# Write output
for line in result:
    print(line)