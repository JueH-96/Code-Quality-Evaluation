# YOUR CODE HERE
def min_button_presses(S):
    count = 0
    i = 0
    while i < len(S):
        if S[i] == '0' and i + 1 < len(S) and S[i+1] == '0':
            count += 1
            i += 2
        else:
            count += 1
            i += 1
    return count

# Read input
S = input().strip()

# Calculate and print the result
result = min_button_presses(S)
print(result)