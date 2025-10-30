# Read the input string
S = input()
N = len(S)
count = 0

# Iterate for idx_i (0-based index for the first character 'A')
for i_idx in range(N):
    # Check if the character at S[i_idx] is 'A'
    if S[i_idx] != 'A':
        continue  # Move to the next i_idx if not 'A'
    
    # Iterate for idx_j (0-based index for the second character 'B')
    # idx_j must be greater than i_idx
    for j_idx in range(i_idx + 1, N):
        # Check if the character at S[j_idx] is 'B'
        if S[j_idx] != 'B':
            continue  # Move to the next j_idx if not 'B'
            
        # Now we have S[i_idx] == 'A' and S[j_idx] == 'B'.
        # We need to determine idx_k based on the equal interval condition.
        # The interval is diff = idx_j - i_idx.
        # Since idx_j > i_idx, diff >= 1.
        diff = j_idx - i_idx
        
        # Calculate idx_k: idx_k = idx_j + diff
        k_idx = j_idx + diff
        
        # Check if idx_k is within the string bounds (idx_k < N).
        # This also implies idx_k > j_idx, satisfying i_idx < j_idx < k_idx.
        if k_idx < N:
            # Check if the character at S[k_idx] is 'C'
            if S[k_idx] == 'C':
                count += 1 # All conditions met, increment count
                
# Print the final count
print(count)