# YOUR CODE HERE
import sys

def main():
    numbers = []
    
    # Read input until EOF
    for line in sys.stdin:
        numbers.append(int(line.strip()))
    
    # Print the numbers in reverse order
    for number in reversed(numbers):
        print(number)

if __name__ == "__main__":
    main()