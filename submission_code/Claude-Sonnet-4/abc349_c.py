# YOUR CODE HERE
S = input().strip()
T = input().strip()

def can_form_subsequence(s, target):
    """Check if target can be formed as a subsequence of s (case insensitive)"""
    target_lower = target.lower()
    i = 0  # pointer for target
    
    for char in s:
        if i < len(target_lower) and char == target_lower[i]:
            i += 1
    
    return i == len(target_lower)

# Method 2: Check if T ends with 'X' and first 2 chars can be formed from S
if T[2] == 'X':
    if can_form_subsequence(S, T[:2]):
        print("Yes")
        exit()

# Method 1: Check if all 3 chars of T can be formed from S
if can_form_subsequence(S, T):
    print("Yes")
else:
    print("No")