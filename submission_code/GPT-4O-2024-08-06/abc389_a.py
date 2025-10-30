# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Extract the first and third characters and convert them to integers
    num1 = int(S[0])
    num2 = int(S[2])
    
    # Calculate the product
    product = num1 * num2
    
    # Print the result
    print(product)

if __name__ == "__main__":
    main()