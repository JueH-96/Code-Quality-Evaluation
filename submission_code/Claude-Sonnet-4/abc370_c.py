# YOUR CODE HERE
S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    result = []
    current = list(S)
    
    while ''.join(current) != T:
        # Find all positions where current differs from T
        diff_positions = []
        for i in range(len(current)):
            if current[i] != T[i]:
                diff_positions.append(i)
        
        # Try changing each differing position and find the lexicographically smallest result
        best_option = None
        best_string = None
        
        for pos in diff_positions:
            # Create a copy and change position pos to match T
            temp = current[:]
            temp[pos] = T[pos]
            temp_string = ''.join(temp)
            
            if best_string is None or temp_string < best_string:
                best_string = temp_string
                best_option = pos
        
        # Apply the best change
        current[best_option] = T[best_option]
        result.append(''.join(current))
    
    print(len(result))
    for s in result:
        print(s)