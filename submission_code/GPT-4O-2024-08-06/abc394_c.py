# YOUR CODE HERE
def replace_wa_with_ac(s):
    # We will use a list to efficiently build the resulting string
    result = []
    i = 0
    while i < len(s):
        # Check if the current and next character form "WA"
        if i < len(s) - 1 and s[i] == 'W' and s[i + 1] == 'A':
            # Replace "WA" with "AC"
            result.append('A')
            result.append('C')
            # Skip the next character as it is part of "WA"
            i += 2
        else:
            # Otherwise, just add the current character to the result
            result.append(s[i])
            i += 1
    # Join the list into a string and return
    return ''.join(result)

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    s = input().strip()
    print(replace_wa_with_ac(s))