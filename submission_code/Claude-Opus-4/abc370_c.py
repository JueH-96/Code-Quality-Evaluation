# YOUR CODE HERE
S = input().strip()
T = input().strip()

# Find positions where S and T differ
diff_positions = []
for i in range(len(S)):
    if S[i] != T[i]:
        diff_positions.append(i)

# Number of changes needed
M = len(diff_positions)
print(M)

# Current string (as a list for easier manipulation)
current = list(S)

# For each step, find the best position to change
for step in range(M):
    best_pos = -1
    best_string = None
    
    # Try changing each remaining position
    for pos in diff_positions:
        if current[pos] != T[pos]:  # This position still needs to be changed
            # Create a copy and change this position
            temp = current[:]
            temp[pos] = T[pos]
            temp_str = ''.join(temp)
            
            # Check if this is the best option so far
            if best_string is None or temp_str < best_string:
                best_string = temp_str
                best_pos = pos
    
    # Apply the best change
    current[best_pos] = T[best_pos]
    print(''.join(current))