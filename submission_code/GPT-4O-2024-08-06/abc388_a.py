# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # The first character of S
    first_char = S[0]
    
    # Concatenate the first character with "UPC"
    result = first_char + "UPC"
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()