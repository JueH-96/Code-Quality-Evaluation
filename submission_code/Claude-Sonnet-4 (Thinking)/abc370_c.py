S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    X = []
    current = list(S)
    target = list(T)
    
    while current != target:
        # Find all positions where current differs from target
        diff_positions = [i for i in range(len(current)) if current[i] != target[i]]
        
        # Try changing each differing position and find the lexicographically smallest result
        best_string = None
        best_position = None
        
        for pos in diff_positions:
            # Create a copy and change the character at position pos
            temp = current[:]
            temp[pos] = target[pos]
            temp_string = ''.join(temp)
            
            if best_string is None or temp_string < best_string:
                best_string = temp_string
                best_position = pos
        
        # Apply the best change
        current[best_position] = target[best_position]
        X.append(''.join(current))
    
    print(len(X))
    for s in X:
        print(s)