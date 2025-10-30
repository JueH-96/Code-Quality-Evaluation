def main():
    # Read the input as a string to easily extract digits
    N_str = input().strip()
    
    # Extract digits as integers
    a = int(N_str[0])  # Hundreds digit
    b = int(N_str[1])  # Tens digit
    c = int(N_str[2])  # Ones digit
    
    # Form the two numbers by rearranging b, c, a and c, a, b
    first_number = 100 * b + 10 * c + a
    second_number = 100 * c + 10 * a + b
    
    # Print the results separated by a space
    print(first_number, second_number)

# Do not forget to call the main function
main()