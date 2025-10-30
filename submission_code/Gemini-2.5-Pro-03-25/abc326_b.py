# YOUR CODE HERE
import sys

# Read input N from standard input
n = int(sys.stdin.readline())

# Start checking numbers from N upwards
current_num = n

while True:
    # Extract the hundreds, tens, and ones digits using integer arithmetic.
    # Since 100 <= N <= 919 and the problem guarantees a solution exists,
    # and the largest 326-like number is 919, the loop will find a 
    # 3-digit solution before current_num reaches 1000.
    # We can safely assume current_num remains a 3-digit number
    # until the solution is found.
    
    hundreds = current_num // 100
    tens = (current_num % 100) // 10
    ones = current_num % 10

    # Check the condition for a 326-like number:
    # product of hundreds and tens digits equals the ones digit.
    if hundreds * tens == ones:
        # If the condition is met, this is the smallest 326-like number >= N.
        # Print the number to standard output.
        print(current_num)
        # Exit the loop as we have found the answer.
        break

    # If the current number is not 326-like, increment and check the next number.
    current_num += 1