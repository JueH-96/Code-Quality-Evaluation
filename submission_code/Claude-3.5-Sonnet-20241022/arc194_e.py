def can_transform(N, X, Y, S, T):
    # Convert strings to lists for easier manipulation
    S = list(S)
    T = list(T)
    
    # Try all possible operations until no more changes are possible
    while True:
        changed = False
        
        # Try Operation A at each possible position
        for i in range(N - (X + Y) + 1):
            # Check if we can apply Operation A
            can_apply_a = True
            for j in range(X):
                if S[i + j] != '0':
                    can_apply_a = False
                    break
            for j in range(Y):
                if S[i + X + j] != '1':
                    can_apply_a = False
                    break
                    
            if can_apply_a:
                # Apply Operation A
                new_S = S.copy()
                for j in range(Y):
                    new_S[i + j] = '1'
                for j in range(X):
                    new_S[i + Y + j] = '0'
                    
                # If this brings us closer to T, apply the change
                if sum(1 for k in range(N) if new_S[k] == T[k]) > sum(1 for k in range(N) if S[k] == T[k]):
                    S = new_S
                    changed = True
        
        # Try Operation B at each possible position
        for i in range(N - (X + Y) + 1):
            # Check if we can apply Operation B
            can_apply_b = True
            for j in range(Y):
                if S[i + j] != '1':
                    can_apply_b = False
                    break
            for j in range(X):
                if S[i + Y + j] != '0':
                    can_apply_b = False
                    break
                    
            if can_apply_b:
                # Apply Operation B
                new_S = S.copy()
                for j in range(X):
                    new_S[i + j] = '0'
                for j in range(Y):
                    new_S[i + X + j] = '1'
                    
                # If this brings us closer to T, apply the change
                if sum(1 for k in range(N) if new_S[k] == T[k]) > sum(1 for k in range(N) if S[k] == T[k]):
                    S = new_S
                    changed = True
        
        # If no changes were made in this iteration, break
        if not changed:
            break
    
    # Check if S equals T
    return ''.join(S) == ''.join(T)

# Read input
N, X, Y = map(int, input().split())
S = input().strip()
T = input().strip()

# Print result
print("Yes" if can_transform(N, X, Y, S, T) else "No")