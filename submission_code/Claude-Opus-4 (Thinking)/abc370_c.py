# YOUR CODE HERE
S = input().strip()
T = input().strip()

# Find positions where S and T differ
diff_positions = []
for i in range(len(S)):
    if S[i] != T[i]:
        diff_positions.append(i)

# Array to store the result
X = []

# Current string (we'll modify it)
current = list(S)

# While there are positions to change
while diff_positions:
    # Find the position to change that gives the lexicographically smallest result
    best_pos = None
    best_string = None
    
    for pos in diff_positions:
        # Create a copy and change this position
        temp = current[:]
        temp[pos] = T[pos]
        temp_str = ''.join(temp)
        
        # Check if this is the best so far
        if best_string is None or temp_str < best_string:
            best_string = temp_str
            best_pos = pos
    
    # Update the current string
    current[best_pos] = T[best_pos]
    X.append(''.join(current))
    
    # Remove this position from the list
    diff_positions.remove(best_pos)

# Output
print(len(X))
for s in X:
    print(s)