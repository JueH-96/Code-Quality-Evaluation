# YOUR CODE HERE
S = input().strip()

presses = 0
i = 0

while i < len(S):
    if i < len(S) - 1 and S[i] == '0' and S[i+1] == '0':
        # Found two consecutive zeros, use "00" button
        presses += 1
        i += 2
    else:
        # Single digit, use corresponding single digit button
        presses += 1
        i += 1

print(presses)