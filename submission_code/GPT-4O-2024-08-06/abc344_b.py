# YOUR CODE HERE
import sys

def main():
    # Read all input lines
    lines = sys.stdin.read().strip().split()
    
    # Convert each line to an integer
    numbers = list(map(int, lines))
    
    # Reverse the list of numbers
    numbers.reverse()
    
    # Print each number on a new line
    for number in numbers:
        print(number)

if __name__ == "__main__":
    main()