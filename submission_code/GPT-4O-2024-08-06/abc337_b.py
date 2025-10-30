# YOUR CODE HERE
def is_extended_abc_string(s):
    # Initialize the expected character
    expected_char = 'A'
    
    # Iterate over the string
    for char in s:
        if char == expected_char:
            continue
        elif expected_char == 'A' and char == 'B':
            expected_char = 'B'
        elif expected_char == 'B' and char == 'C':
            expected_char = 'C'
        else:
            # If the character does not match the expected sequence, return "No"
            return "No"
    
    # If we have successfully iterated through the string, return "Yes"
    return "Yes"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    s = input().strip()
    print(is_extended_abc_string(s))