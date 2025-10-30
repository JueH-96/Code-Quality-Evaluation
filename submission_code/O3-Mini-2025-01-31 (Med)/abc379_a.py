def main():
    # Read the input number as a string to easily access each digit
    N = input().strip()
    
    # Extract digits (a, b, c)
    a, b, c = N[0], N[1], N[2]
    
    # Form new integers according to the problem statement
    num1 = int(b + c + a)  # Arrange b, c, a
    num2 = int(c + a + b)  # Arrange c, a, b
    
    # Print the result separated by a space
    print(num1, num2)

# Call the main function
if __name__ == '__main__':
    main()