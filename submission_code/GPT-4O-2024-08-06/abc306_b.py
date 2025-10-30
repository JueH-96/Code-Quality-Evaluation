# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    # Read the input
    A = list(map(int, input().strip().split()))
    
    # Calculate the result
    result = 0
    for i in range(64):
        result += A[i] * (2 ** i)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()