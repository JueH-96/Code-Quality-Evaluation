# YOUR CODE HERE
N = int(input())

if N <= 999:
    print(N)
else:
    # Convert to string to work with digits
    s = str(N)
    num_digits = len(s)
    
    # Keep first 3 digits, replace rest with zeros
    result = s[:3] + '0' * (num_digits - 3)
    print(int(result))