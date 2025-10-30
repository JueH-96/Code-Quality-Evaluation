# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    # Parse the input
    A, B = map(int, data.split())
    
    # Calculate A^B and B^A
    result = (A ** B) + (B ** A)
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()