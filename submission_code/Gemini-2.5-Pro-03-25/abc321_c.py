import sys

# Set a reasonable recursion depth limit. 
# The maximum recursion depth for this problem is 10 (for the number 9876543210),
# which is well within typical default limits. Explicitly setting it is usually
# not necessary for this problem but can be good practice in competitive programming.
# sys.setrecursionlimit(2000) 

# Global list to store all generated 321-like numbers
all_321_like_numbers = []

def generate(current_num):
    """
    Recursively generates 321-like numbers.
    A positive integer is 321-like if its digits are strictly decreasing from left to right.
    
    This function first adds the `current_num` (which is assumed to be 321-like) 
    to the global list `all_321_like_numbers`.
    
    Then, it finds the last digit of `current_num`. It iterates through all possible digits `d` 
    that are strictly smaller than the last digit (i.e., `0 <= d < last_digit`). 
    For each such `d`, it forms a new number by appending `d` to `current_num` 
    (effectively `current_num * 10 + d`) and makes a recursive call with this new number.
    
    The base case for the recursion is when the last digit is 0, as no non-negative digit `d` 
    satisfies `d < 0`. The loop `for d in range(last_digit)` handles this naturally because `range(0)` is empty.
    """
    # Add the current 321-like number to our list.
    all_321_like_numbers.append(current_num)
    
    # Get the last digit of the current number.
    last_digit = current_num % 10
    
    # If the last digit is 0, we cannot append any digit d such that d < 0.
    # The loop condition below handles this automatically.
    
    # Iterate through all possible digits `d` that can be appended.
    # `d` must be strictly smaller than the `last_digit` to maintain the 321-like property.
    # The possible values for `d` are integers from 0 up to `last_digit - 1`.
    for d in range(last_digit): 
        # Construct the new number by appending digit `d`.
        new_num = current_num * 10 + d
        # Make a recursive call to continue generating numbers starting with `new_num`.
        generate(new_num)

# Main part of the script execution starts here.

# The generation process must start from some initial numbers.
# By definition, all single-digit positive integers (1 through 9) are 321-like numbers.
# We use these as the starting points for our recursive generation function.
for s in range(1, 10):
    generate(s)

# After the `generate` function has been called for all starting digits 1 through 9,
# the `all_321_like_numbers` list contains all possible 321-like numbers.
# However, they are not necessarily in sorted order due to the nature of DFS/recursive generation.
# We need to sort the list to find the K-th smallest number.
all_321_like_numbers.sort()

# Read the integer K from standard input. K specifies which smallest number we need to find.
K = int(sys.stdin.readline())

# The problem asks for the K-th smallest 321-like number.
# Since Python lists are 0-indexed, the K-th element in the sorted list 
# is located at index K-1.
print(all_321_like_numbers[K-1])