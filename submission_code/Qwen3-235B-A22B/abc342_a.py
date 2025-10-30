s = input().strip()

if s[0] == s[1]:
    # Find the first differing character
    for i in range(len(s)):
        if s[i] != s[0]:
            print(i + 1)
            break
else:
    # Check the third character to determine the majority
    if s[2] == s[0]:
        # Majority is s[0], second character is different
        print(2)
    else:
        # Majority is s[1], first character is different
        print(1)