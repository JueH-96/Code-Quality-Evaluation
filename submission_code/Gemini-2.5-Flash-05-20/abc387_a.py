import sys

def main():
    # Read the input line from standard input
    line = sys.stdin.readline().strip()
    
    # Split the line by space to get two strings
    a_str, b_str = line.split()
    
    # Convert the strings to integers
    A = int(a_str)
    B = int(b_str)
    
    # Calculate the sum of A and B
    sum_ab = A + B
    
    # Calculate the square of the sum
    result = sum_ab ** 2
    
    # Print the result to standard output
    print(result)

if __name__ == '__main__':
    main()