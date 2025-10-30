# Read the input string from stdin
S = input()

# Check if the input string is an Extended ABC string
def is_extended_abc(S):
    # Check if the string is empty
    if not S:
        return True
    
    # Initialize the lengths of the Extended A, B, and C strings
    a_len = 0
    b_len = 0
    c_len = 0
    
    # Iterate through the string and count the lengths of the Extended A, B, and C strings
    for c in S:
        if c == 'A':
            a_len += 1
        elif c == 'B':
            b_len += 1
        else:
            c_len += 1
    
    # Check if the lengths of the Extended A, B, and C strings add up to the length of the input string
    return a_len + b_len + c_len == len(S)

# Print the result
if is_extended_abc(S):
    print("Yes")
else:
    print("No")