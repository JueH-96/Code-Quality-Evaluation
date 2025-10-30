def main():
    import sys
    
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # If S and T are already equal, no steps needed
    if S == T:
        print(0)
        return
    
    # Identify the indices where S and T differ
    diff_indices = {i for i in range(len(S)) if S[i] != T[i]}
    
    X = []
    current = S
    
    # Greedily fix one differing position at a time
    while diff_indices:
        best_string = None
        best_index = None
        
        # Try fixing each remaining difference and pick
        # the lexicographically smallest result
        for i in diff_indices:
            candidate = current[:i] + T[i] + current[i+1:]
            if best_string is None or candidate < best_string:
                best_string = candidate
                best_index = i
        
        current = best_string
        X.append(current)
        diff_indices.remove(best_index)
    
    # Output the result
    print(len(X))
    for x in X:
        print(x)

# Do not forget to call main()
main()