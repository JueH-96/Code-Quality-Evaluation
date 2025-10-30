def main():
    import sys
    
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # If already equal, no operations needed
    if S == T:
        print(0)
        return
    
    # We'll build up the transformation array
    X = []
    current = list(S)  # Work with a list for easy single-character updates
    
    while True:
        if ''.join(current) == T:
            break
        
        # Find all mismatches
        mismatches = []
        for i in range(len(current)):
            if current[i] != T[i]:
                mismatches.append(i)
        
        # Among all mismatches, choose the one that yields 
        # the lexicographically smallest next string
        best_string = None
        best_index = None
        
        for i in mismatches:
            saved_char = current[i]
            current[i] = T[i]
            candidate = ''.join(current)
            current[i] = saved_char  # revert
            
            if best_string is None or candidate < best_string:
                best_string = candidate
                best_index = i
        
        # Apply the best fix
        current[best_index] = T[best_index]
        X.append(''.join(current))
        
        if ''.join(current) == T:
            break
    
    # Output
    print(len(X))
    for s in X:
        print(s)

# Don't forget to call main()
main()