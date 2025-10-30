# Read input
S = input().strip()
T = input().strip()

# Initialize variables
s_pos = 0  # Position in S
t_pos = 0  # Position in T
result = []  # Store positions of correctly typed characters

# Process each character in T
while s_pos < len(S) and t_pos < len(T):
    if S[s_pos] == T[t_pos]:
        # Character matches, it's correctly typed
        result.append(t_pos + 1)  # Add 1 because positions are 1-based
        s_pos += 1
    t_pos += 1

# Print result
print(*result)