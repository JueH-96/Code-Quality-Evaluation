# YOUR CODE HERE
N, M = map(int, input().split())
S = list(input().strip())
T = input().strip()

# Convert S to list of integers for easier manipulation
S = [int(ch) for ch in S]

# Process each character in T
for k in range(M):
    new_digit = int(T[k])
    
    # Find the best position to place this digit
    best_pos = -1
    best_gain = float('-inf')
    
    for i in range(N):
        # Calculate the gain from replacing position i
        # Gain is proportional to (new_digit - current_digit) * position_weight
        # Position weight is higher for leftmost positions
        gain = (new_digit - S[i]) * (10 ** (N - i - 1))
        
        if gain > best_gain:
            best_gain = gain
            best_pos = i
    
    # Replace the character at the best position
    S[best_pos] = new_digit

# Convert back to string and print
result = ''.join(map(str, S))
print(result)