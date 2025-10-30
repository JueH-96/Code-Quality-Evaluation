import sys

def main():
    """
    Reads a string from standard input, performs the "ABC" removal operations,
    and prints the final string to standard output.
    """
    # Read the entire input line from stdin and remove trailing newline characters.
    s = sys.stdin.readline().strip()

    # `result_stack` will store the characters of the final string.
    # It's used as a stack to efficiently handle the removal of "ABC" sequences.
    result_stack = []

    # Iterate through each character of the input string.
    for char in s:
        # Add the current character to our stack.
        result_stack.append(char)

        # Check if the stack has at least 3 characters and if the last three form "ABC".
        # This check is only necessary after appending a character, as that is the only
        # time a new "ABC" sequence can be formed at the end.
        if (len(result_stack) >= 3 and
            result_stack[-3] == 'A' and
            result_stack[-2] == 'B' and
            result_stack[-1] == 'C'):
            
            # If the condition is met, remove the last three characters.
            # Popping three times effectively removes the "ABC" sequence.
            result_stack.pop()
            result_stack.pop()
            result_stack.pop()

    # Join the characters remaining in the stack to form the final string
    # and print it to standard output.
    print("".join(result_stack))

if __name__ == "__main__":
    main()