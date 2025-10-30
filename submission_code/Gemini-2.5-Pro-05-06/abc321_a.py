# YOUR CODE HERE
# Read the input N as a string
N_str = input()

# Get the number of digits
d = len(N_str)

# All one-digit positive integers are 321-like Numbers.
if d == 1:
    print("Yes")
else:
    # Assume it's a 321-like number until a violation is found
    is_321_like = True
    
    # Iterate through adjacent digits
    # The loop runs from i = 0 to d-2.
    # This compares (N_str[0], N_str[1]), (N_str[1], N_str[2]), ..., (N_str[d-2], N_str[d-1]).
    for i in range(d - 1):
        # Convert character digits to integers for comparison
        digit_current = int(N_str[i])
        digit_next = int(N_str[i+1])

        # Check if the current digit is strictly greater than the next digit.
        # If not (i.e., current_digit <= next_digit), then it's not a 321-like number.
        if digit_current <= digit_next:
            is_321_like = False
            break  # Violation found, no need to check further
    
    # Print the result based on whether a violation was found
    if is_321_like:
        print("Yes")
    else:
        print("No")