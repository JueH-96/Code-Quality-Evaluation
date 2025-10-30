def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    # The input string is the first token
    s = input_data[0]

    # We'll use a stack-based approach to simulate the removal process.
    # As we iterate through s, we append characters to a list (acting as a stack).
    # After appending, if the last three characters are "ABC", we remove them.
    stack = []
    for char in s:
        stack.append(char)
        # Check if the last 3 characters form the substring "ABC"
        if len(stack) >= 3 and stack[-3] == 'A' and stack[-2] == 'B' and stack[-1] == 'C':
            # Remove last three characters
            stack.pop()
            stack.pop()
            stack.pop()
    
    # Join the stack to form the final string.
    final_string = ''.join(stack)
    sys.stdout.write(final_string)

if __name__ == '__main__':
    main()