# YOUR CODE HERE
import sys

def main():
    # Read the input from standard input
    S = sys.stdin.read().strip()
    
    # Split the string by '.' and get the last element
    last_substring = S.split('.')[-1]
    
    # Print the result
    print(last_substring)

if __name__ == "__main__":
    main()