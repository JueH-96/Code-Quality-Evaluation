# YOUR CODE HERE
import sys

def main():
    # Read input from standard input
    input_value = sys.stdin.read().strip()
    
    # Convert the input string to a float
    number = float(input_value)
    
    # Print the number without trailing zeros
    print(number)

if __name__ == "__main__":
    main()