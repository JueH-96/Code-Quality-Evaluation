def main():
    import sys
    # Read the input string from standard input
    input_str = sys.stdin.read().strip()
    
    # Extract the first and third characters, then convert them to integers
    num1 = int(input_str[0])
    num2 = int(input_str[2])
    
    # Compute the product of the two numbers
    result = num1 * num2
    
    # Output the result
    print(result)

# Call the main function
if __name__ == "__main__":
    main()