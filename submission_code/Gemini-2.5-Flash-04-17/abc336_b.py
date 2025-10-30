# YOUR CODE HERE
# Read the integer N from standard input.
N = int(input())

# Initialize a counter for the number of trailing zeros.
count = 0

# Calculate the number of trailing zeros using a bitwise approach.
# The number of trailing zeros in binary is equivalent to the largest power of 2
# that divides N. This can be found by repeatedly checking the least significant bit (LSB).
# If the LSB is 0, it means N is even and divisible by 2. We increment the count
# and right-shift N (effectively dividing by 2) to check the next bit.
# This process continues until the LSB is 1 (meaning the number is odd).
# The loop condition (N & 1) == 0 checks if the LSB of N is 0.
# N >>= 1 performs a bitwise right shift on N, discarding the LSB.
# Since the problem guarantees N >= 1, N will eventually become an odd number,
# and the loop will terminate.
while (N & 1) == 0:
    N >>= 1
    count += 1

# Print the total count of trailing zeros to standard output.
print(count)
# YOUR CODE HERE