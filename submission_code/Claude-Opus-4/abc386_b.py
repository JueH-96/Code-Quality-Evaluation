# YOUR CODE HERE
S = input().strip()

button_presses = 0
i = 0

while i < len(S):
    if S[i] == '0':
        # Count consecutive zeros
        zero_count = 0
        j = i
        while j < len(S) and S[j] == '0':
            zero_count += 1
            j += 1
        
        # Use "00" button as much as possible
        button_presses += zero_count // 2  # Number of "00" presses
        button_presses += zero_count % 2   # Remaining single "0" press if odd
        
        i = j  # Skip all the zeros we just processed
    else:
        # For non-zero digits, we must press the single digit button
        button_presses += 1
        i += 1

print(button_presses)