# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    # Split the input into two integers A and B
    A, B = map(int, data.split())
    
    # Calculate the square of the sum of A and B
    result = (A + B) ** 2
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()