def is_extended_abc_string(s):
    # Check if the string is empty or has only one type of character
    if not s or s == len(s) * s[0]:
        return True

    # Check if the string has the pattern A*B*C*
    for i in range(len(s)):
        if s[i] == 'A':
            continue
        for j in range(i, len(s)):
            if s[j] == 'B':
                continue
            for k in range(j, len(s)):
                if s[k] == 'C':
                    continue
                # If we find a character that is not C after B, return False
                return False
            # If we find a character that is not B after A, return False
            return False
    # If all characters are A followed by B followed by C, return True
    return True

# Read input from stdin
S = input().strip()

# Check if the string is an Extended ABC string and print the result
print("Yes" if is_extended_abc_string(S) else "No")