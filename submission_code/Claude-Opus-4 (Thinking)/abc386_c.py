# YOUR CODE HERE
K = int(input())
S = input()
T = input()

def solve():
    # If strings are equal, no operation needed
    if S == T:
        return "Yes"
    
    # If length difference is more than 1, impossible with 1 operation
    if abs(len(S) - len(T)) > 1:
        return "No"
    
    if len(S) == len(T):
        # Check if they differ in exactly 1 position (replace operation)
        diff_count = sum(1 for i in range(len(S)) if S[i] != T[i])
        return "Yes" if diff_count == 1 else "No"
    
    elif len(S) == len(T) - 1:
        # Check if we can insert one character in S to make it T
        i = j = 0
        while i < len(S) and j < len(T):
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                j += 1
                break
        # After the first mismatch, the rest should match
        while i < len(S) and j < len(T):
            if S[i] != T[j]:
                return "No"
            i += 1
            j += 1
        return "Yes" if i == len(S) and j == len(T) else "No"
    
    else:  # len(S) == len(T) + 1
        # Check if we can delete one character from S to make it T
        i = j = 0
        while i < len(S) and j < len(T):
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                i += 1
                break
        # After the first mismatch, the rest should match
        while i < len(S) and j < len(T):
            if S[i] != T[j]:
                return "No"
            i += 1
            j += 1
        return "Yes" if i == len(S) and j == len(T) else "No"

print(solve())