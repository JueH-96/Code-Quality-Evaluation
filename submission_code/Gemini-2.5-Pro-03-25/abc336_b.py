# YOUR CODE HERE
import sys

def main():
    # Read integer N from standard input.
    # sys.stdin.readline() reads a line including the newline character.
    # int() correctly parses the integer, ignoring trailing whitespace like the newline.
    n = int(sys.stdin.readline())
    
    # Initialize count of trailing zeros
    count = 0
    
    # The problem states N is a positive integer (N >= 1).
    # The number of trailing zeros in the binary representation of N is
    # the highest power of 2 that divides N.
    # We can find this by repeatedly dividing N by 2 as long as it is even,
    # and counting how many times we can do this.
    
    # We use bitwise operations for efficiency:
    # (n & 1) checks the last bit. If it's 0, the number is even.
    # n >>= 1 performs integer division by 2 (right bit shift).
    
    # The loop continues as long as N is even.
    # When N becomes odd, its binary representation ends in 1,
    # meaning (n & 1) will be 1, and the condition (n & 1) == 0 becomes false,
    # terminating the loop.
    while (n & 1) == 0:
        n >>= 1 
        count += 1
    
    # Print the final count to standard output.
    # The print function automatically adds a newline at the end.
    print(count)

# Standard boilerplate to call the main function when the script is executed.
if __name__ == '__main__':
    main()