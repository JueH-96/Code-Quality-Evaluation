# Read input
K = int(input())
S = input()
T = input()

def can_transform(S, T):
    # If strings are equal, no operations needed
    if S == T:
        return True
    
    # If length difference is more than 1, impossible with one operation
    if abs(len(S) - len(T)) > 1:
        return False
    
    # Case 1: Same length - possible with one replacement
    if len(S) == len(T):
        differences = sum(1 for i in range(len(S)) if S[i] != T[i])
        return differences <= 1
    
    # Case 2: T is one character longer than S - possible with one insertion
    elif len(T) == len(S) + 1:
        i = j = 0
        differences = 0
        while i < len(S) and j < len(T):
            if S[i] != T[j]:
                differences += 1
                j += 1
                continue
            i += 1
            j += 1
        return differences <= 1
    
    # Case 3: S is one character longer than T - possible with one deletion
    elif len(S) == len(T) + 1:
        i = j = 0
        differences = 0
        while i < len(S) and j < len(T):
            if S[i] != T[j]:
                differences += 1
                i += 1
                continue
            i += 1
            j += 1
        return differences <= 1
    
    return False

# Output result
print("Yes" if can_transform(S, T) else "No")