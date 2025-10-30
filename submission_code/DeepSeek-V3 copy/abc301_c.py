from collections import defaultdict

def can_win(S, T):
    # Define the set of characters that can replace '@'
    replaceable = {'a', 't', 'c', 'o', 'd', 'e', 'r'}
    
    # Count the frequency of each character in S and T, excluding '@'
    count_S = defaultdict(int)
    count_T = defaultdict(int)
    
    for char in S:
        if char != '@':
            count_S[char] += 1
    
    for char in T:
        if char != '@':
            count_T[char] += 1
    
    # Calculate the number of '@' in S and T
    at_S = S.count('@')
    at_T = T.count('@')
    
    # For each character in the replaceable set, calculate the difference in counts
    # and adjust the number of '@' accordingly
    for char in replaceable:
        diff = count_S[char] - count_T[char]
        if diff > 0:
            at_T -= diff
        elif diff < 0:
            at_S += diff
    
    # Now, handle the remaining characters that are not in the replaceable set
    # These must have the same count in both S and T
    for char in count_S:
        if char not in replaceable and char != '@':
            if count_S[char] != count_T[char]:
                return False
    
    for char in count_T:
        if char not in replaceable and char != '@':
            if count_T[char] != count_S[char]:
                return False
    
    # Finally, the number of '@' in S and T must be non-negative
    if at_S < 0 or at_T < 0:
        return False
    
    return True

# Read input
S = input().strip()
T = input().strip()

# Determine if it's possible to win
if can_win(S, T):
    print("Yes")
else:
    print("No")