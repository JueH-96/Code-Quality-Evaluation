# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    N = int(input().strip())
    
    # Initialize an empty result string
    result = ""
    
    # We need to alternate between 1 and 0, starting with 1
    for i in range(N):
        result += "10"
    
    # Add the last '1' to make it N+1 ones
    result += "1"
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()