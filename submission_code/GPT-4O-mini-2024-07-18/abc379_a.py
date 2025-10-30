# YOUR CODE HERE
def rearrange_digits(N):
    # Convert the integer to a string to access individual digits
    str_N = str(N)
    
    # Extract digits a, b, c
    a = str_N[0]  # hundreds
    b = str_N[1]  # tens
    c = str_N[2]  # ones
    
    # Form the new integers
    first_number = int(b + c + a)  # bca
    second_number = int(c + a + b)  # cab
    
    # Print the results
    print(first_number, second_number)

# Read input
N = int(input().strip())
rearrange_digits(N)