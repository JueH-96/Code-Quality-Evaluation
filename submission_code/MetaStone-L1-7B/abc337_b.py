s = input().strip()

a_end = 0
while a_end < len(s) and s[a_end] == 'A':
    a_end += 1

if a_end == len(s):
    # All characters are 'A's
    print("Yes")
elif a_end == 0:
    # Check if all characters are 'A's
    all_a = True
    for c in s:
        if c != 'A':
            all_a = False
            break
    if all_a:
        print("Yes")
    else:
        # Proceed to check if the rest are B followed by C
        b_end = 0
        while b_end < len(s) and s[b_end] == 'B':
            b_end += 1
        valid = True
        for i in range(b_end, len(s)):
            if s[i] != 'C':
                valid = False
                break
        print("Yes" if valid else "No")
else:
    # Proceed to find the B transition point
    b_end = a_end
    while b_end < len(s) and s[b_end] == 'B':
        b_end += 1

    # Check if from a_end to b_end are all B's
    valid = True
    for i in range(a_end, b_end):
        if s[i] != 'B':
            valid = False
            break
    if not valid:
        print("No")
    else:
        # Check from b_end to end are all C's
        valid = True
        for i in range(b_end, len(s)):
            if s[i] != 'C':
                valid = False
                break
        print("Yes" if valid else "No")