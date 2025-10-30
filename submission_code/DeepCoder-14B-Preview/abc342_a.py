s = input().strip()

if s[0] == s[1]:
    # Majority is s[0], find the unique character
    for i in range(len(s)):
        if s[i] != s[0]:
            print(i + 1)
            exit()
else:
    # Check the third character to determine the majority
    if s[2] == s[0]:
        # s[1] is the unique character
        print(2)
    else:
        # s[0] is the unique character
        print(1)