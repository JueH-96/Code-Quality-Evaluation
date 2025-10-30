def main():
    # Read the integer N from input
    N = int(input().strip())
    
    # Build the result string based on whether the index i is a multiple of 3
    result = []
    for i in range(1, N + 1):
        if i % 3 == 0:
            result.append('x')
        else:
            result.append('o')
    
    # Print the result as a concatenated string
    print("".join(result))

# Call the main function
main()