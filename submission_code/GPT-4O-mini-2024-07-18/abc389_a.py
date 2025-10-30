# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Extract the digits from the string
    first_digit = int(S[0])
    second_digit = int(S[2])
    
    # Calculate the product
    product = first_digit * second_digit
    
    # Print the result
    print(product)

if __name__ == "__main__":
    main()