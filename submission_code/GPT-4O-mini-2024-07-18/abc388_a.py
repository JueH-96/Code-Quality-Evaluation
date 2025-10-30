# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Get the first character of S
    first_character = S[0]
    
    # Concatenate the first character with "UPC"
    result = first_character + "UPC"
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()