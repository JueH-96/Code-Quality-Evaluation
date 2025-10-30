# YOUR CODE HERE
S = input().strip()

can_see_A = True
can_see_B = True
valid = True

for char in S:
    if char == 'A':
        if not can_see_A:
            valid = False
            break
    elif char == 'B':
        if not can_see_B:
            valid = False
            break
        can_see_A = False
    elif char == 'C':
        can_see_A = False
        can_see_B = False

if valid:
    print("Yes")
else:
    print("No")