# YOUR CODE HERE
def is_airport_code(s, t):
    # Check if t can be formed by a subsequence of length 3 from s
    i = 0
    for char in s:
        if i < 3 and char.upper() == t[i]:
            i += 1
    if i == 3:
        return True

    # Check if t can be formed by a subsequence of length 2 from s and appending 'X'
    i = 0
    for char in s:
        if i < 2 and char.upper() == t[i]:
            i += 1
    if i == 2 and t[2] == 'X':
        return True

    return False

# Read input
s = input().strip()
t = input().strip()

# Determine if t is an airport code for s
if is_airport_code(s, t):
    print("Yes")
else:
    print("No")