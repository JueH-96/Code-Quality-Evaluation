import sys

all_321_numbers = []

def dfs(current_num, last_digit):
    """
    Recursively generates 321-like numbers.
    :param current_num: The number built so far.
    :param last_digit: The last digit appended to current_num.
    """
    # Add the current number (which is a 321-like number) to the list
    all_321_numbers.append(current_num)
    
    # Try to append a new digit. This new digit must be strictly smaller than last_digit.
    # It can range from 0 up to last_digit - 1.
    # The order of iteration for next_digit_val (e.g., 0 to last_digit-1, or last_digit-1 down to 0)
    # does not affect the final set of generated numbers, as the list will be sorted globally later.
    for next_digit_val in range(last_digit):  # Iterates next_digit_val from 0, 1, ..., last_digit-1
        new_num = current_num * 10 + next_digit_val
        dfs(new_num, next_digit_val)

# Initial calls to start the generation process:
# A 321-like number must be a positive integer.
# The first digit can be any digit from 1 to 9.
# For each such starting digit `i`, `i` itself is a 1-digit 321-like number.
# The `last_digit` for this initial call is `i`.
for i in range(1, 10):  # i will be 1, 2, ..., 9
    dfs(i, i)

# After the recursive generation, `all_321_numbers` contains all 321-like numbers.
# Sort them to find the K-th smallest.
all_321_numbers.sort()

# Read K from standard input.
K = int(sys.stdin.readline())

# Print the K-th smallest 321-like number.
# K is 1-indexed according to the problem, while Python lists are 0-indexed.
# So, we access the element at index K-1.
print(all_321_numbers[K-1])