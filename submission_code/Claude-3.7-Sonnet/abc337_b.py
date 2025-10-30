# YOUR CODE HERE
s = input().strip()

# Initialize state variables
seen_b = False
seen_c = False
is_valid = True

for char in s:
    if char == 'A':
        # If we've already seen B or C, then A is out of place
        if seen_b or seen_c:
            is_valid = False
            break
    elif char == 'B':
        # Mark that we've seen B
        seen_b = True
        # If we've already seen C, then B is out of place
        if seen_c:
            is_valid = False
            break
    elif char == 'C':
        # Mark that we've seen C
        seen_c = True

if is_valid:
    print("Yes")
else:
    print("No")