import sys

# Read the input N
n = int(sys.stdin.readline())

# The N-th good integer corresponds to the (N-1)-th non-negative integer
# represented in base 5 using the digits {0, 2, 4, 6, 8}.
# The mapping from base-5 digits {0, 1, 2, 3, 4} to good digits {0, 2, 4, 6, 8}
# is simply digit * 2.

# We work with N-1 because the 1st good integer (0) corresponds to N=1, which is index 0.
num = n - 1

# Special case for N=1, where N-1 = 0.
# The base-5 representation of 0 is '0'. The mapped digit is 2*0=0.
# The result is 0.
# The loop below handles num > 0. We need to explicitly handle num == 0.
if num == 0:
    print(0)
else:
    # Convert N-1 to base 5 and map the digits
    mapped_digits_reversed = []
    while num > 0:
        # Get the least significant digit in base 5
        remainder = num % 5
        # Map the digit to an even digit (0->0, 1->2, 2->4, 3->6, 4->8)
        mapped_digit = 2 * remainder
        # Append the mapped digit (as string) to the list
        mapped_digits_reversed.append(str(mapped_digit))
        # Move to the next digit (integer division by 5)
        num //= 5

    # The mapped digits are collected from least significant to most significant
    # (right to left in base 5).
    # We need to reverse the list to get them in the correct order (most significant first).
    mapped_digits_reversed.reverse()

    # Join the string digits to form the final result
    result = "".join(mapped_digits_reversed)

    # Print the result
    print(result)