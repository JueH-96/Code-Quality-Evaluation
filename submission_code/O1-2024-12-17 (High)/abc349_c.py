def is_subsequence(s, t):
    """Check if t is a subsequence of s."""
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            j += 1
        i += 1
    return j == len(t)

def main():
    S = input().strip()
    T = input().strip()
    
    # Case 1: T is formed by a subsequence of length 3
    if is_subsequence(S, T.lower()):
        print("Yes")
        return
    
    # Case 2: T is formed by a subsequence of length 2 plus 'X'
    if T[2] == 'X':
        if is_subsequence(S, T[:2].lower()):
            print("Yes")
            return
    
    # If neither case is satisfied
    print("No")

main()