import sys

# Create an empty list to store the input numbers
numbers = []

# Loop indefinitely until the termination condition is met
while True:
    try:
        # Read a line from standard input.
        # input() reads the line and automatically removes the trailing newline.
        line = input()
        
        # Convert the read line (string) to an integer.
        # This might raise ValueError if the input is not a valid integer,
        # but the problem constraints guarantee valid integer input.
        num = int(line)
        
        # Append the integer to our list
        numbers.append(num)
        
        # Check if the number read is 0.
        # According to the problem statement, 0 marks the end of the sequence.
        if num == 0:
            break # Exit the loop if the number is 0
            
    except EOFError:
        # This handles the case where the input stream ends unexpectedly
        # before a 0 is found. While the problem guarantees a 0 terminator,
        # this makes the code slightly more robust.
        break
    # We don't explicitly handle ValueError here because the problem constraints
    # guarantee that inputs are integers.

# Now, the `numbers` list contains all the integers read in the order A_1, A_2, ..., A_N (where A_N = 0).
# We need to print them in reverse order: A_N, A_{N-1}, ..., A_1.

# We can iterate through the list in reverse using slicing `[::-1]`.
# This creates a reversed copy of the list.
for num in numbers[::-1]:
    # Print each number. By default, print() adds a newline character at the end.
    print(num)

# Alternative ways to print in reverse (for illustration):

# 1. Using reversed() built-in function (returns an iterator):
# for num in reversed(numbers):
#     print(num)

# 2. Using index manipulation:
# n = len(numbers)
# for i in range(n - 1, -1, -1):
#     print(numbers[i])