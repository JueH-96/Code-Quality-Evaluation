# YOUR CODE HERE
# Read the input integer N
N = int(input())

# Start checking from N
current_number = N

# Loop until a 326-like number is found
while True:
    # Extract the digits of the current number
    # Hundreds digit
    hundreds = current_number // 100
    # Tens digit
    tens = (current_number % 100) // 10
    # Ones digit
    ones = current_number % 10

    # Check if the product of hundreds and tens digits equals the ones digit
    if hundreds * tens == ones:
        # If it is a 326-like number, print it and stop
        print(current_number)
        break
    else:
        # If not, check the next integer
        current_number += 1