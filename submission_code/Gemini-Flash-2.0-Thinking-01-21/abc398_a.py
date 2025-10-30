import sys

# Read the input N
N = int(sys.stdin.readline())

# Create a list of N hyphens initially
result_list = ['-' for _ in range(N)]

# Check if N is odd or even to determine the position(s) of '='
if N % 2 == 1:
    # If N is odd, there is exactly one central character.
    # The single '=' must be at this central position to satisfy palindrome condition
    # and the "exactly one =" condition.
    center_index = N // 2
    result_list[center_index] = '='
else:
    # If N is even, there are two central positions.
    # The two adjacent '='s must be at these two central positions
    # to satisfy palindrome condition and the "exactly two adjacent =" condition.
    center_right_index = N // 2
    center_left_index = center_right_index - 1
    result_list[center_left_index] = '='
    result_list[center_right_index] = '='

# Join the list of characters into a string
result_string = ''.join(result_list)

# Print the resulting string
print(result_string)