# Read the input integer N
N = int(input())

# Initialize a counter for the number of trailing zeros
ctz_count = 0

# Loop as long as N is an even number.
# An even number ends with a '0' in its binary representation.
# An odd number ends with a '1' in its binary representation.
# The loop stops when N becomes odd, meaning we have found the first '1' from the right.
while N % 2 == 0:
    # If N is even, it means it has at least one trailing zero.
    # Divide N by 2 (integer division) which is equivalent to a right bit shift.
    # This effectively removes the rightmost '0'.
    N //= 2
    
    # Increment the counter for each trailing zero found.
    ctz_count += 1

# Print the final count of consecutive zeros at the end of N's binary notation
print(ctz_count)