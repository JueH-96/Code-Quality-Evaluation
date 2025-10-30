def main():
    import sys

    S = sys.stdin.readline().rstrip('
')
    T = sys.stdin.readline().rstrip('
')

    # If they are already equal, answer is 0 and no steps
    if S == T:
        print(0)
        return
    
    # Identify mismatch positions
    mismatch_positions = [i for i in range(len(S)) if S[i] != T[i]]
    
    # We will build the resulting array X
    X = []
    current = S
    
    # While there are still mismatches, pick the change that yields
    # the lexicographically smallest next string.
    while mismatch_positions:
        candidates = []
        for pos in mismatch_positions:
            # Create a candidate next string by fixing one mismatch
            candidate = current[:pos] + T[pos] + current[pos+1:]
            candidates.append((candidate, pos))
        
        # Pick the lexicographically smallest candidate
        best_string, best_pos = min(candidates)
        
        # Add it to our sequence
        X.append(best_string)
        
        # Update current string
        current = best_string
        
        # Remove this mismatch from the list
        mismatch_positions.remove(best_pos)
    
    # Print the result
    print(len(X))
    for s in X:
        print(s)

# Do not forget to call main!
if __name__ == "__main__":
    main()