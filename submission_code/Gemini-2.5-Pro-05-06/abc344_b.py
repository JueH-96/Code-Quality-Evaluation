# YOUR CODE HERE
numbers = []
while True:
    # Read a line from standard input
    s = input()
    # Convert the string to an integer
    num = int(s)
    # Add the integer to our list
    numbers.append(num)
    # If the number is 0, it's A_N, so we stop reading
    if num == 0:
        break

# Iterate through the list in reverse order and print each number.
# reversed() returns an iterator that yields elements from the list
# in reverse order.
for val in reversed(numbers):
    # print() by default prints its argument followed by a newline character.
    print(val)