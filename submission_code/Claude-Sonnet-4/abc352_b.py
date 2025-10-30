# YOUR CODE HERE
S = input().strip()
T = input().strip()

positions = []
t_index = 0

for char in S:
    # Find the next occurrence of char in T starting from t_index
    while t_index < len(T) and T[t_index] != char:
        t_index += 1
    
    # Record the position (1-indexed)
    positions.append(t_index + 1)
    
    # Move to the next character in T
    t_index += 1

print(' '.join(map(str, positions)))