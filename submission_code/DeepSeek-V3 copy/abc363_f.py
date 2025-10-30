# YOUR CODE HERE
def find_palindrome_formula(N):
    # Since the string must be a palindrome and start with a digit, we can try to construct it
    # by iterating over possible lengths and digits.
    # The maximum length is 1000, but for practical purposes, we can limit it to a smaller number.
    # We will try to find a string S such that eval(S) == N and S is a palindrome.
    
    # We will try to construct S by considering the first half and mirroring it.
    # The first character must be a digit, and the rest can be digits or '*'.
    
    # We will limit the length to 20 for practical purposes, but it can be increased if needed.
    max_length = 20
    
    for length in range(1, max_length + 1):
        # Determine the first half of the string
        half_length = (length + 1) // 2
        # Generate all possible combinations for the first half
        # The first character must be a digit
        from itertools import product
        digits = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
        symbols = digits + ['*']
        for first_half in product(symbols, repeat=half_length):
            if first_half[0] not in digits:
                continue
            # Construct the full string by mirroring the first half
            if length % 2 == 0:
                full_string = ''.join(first_half) + ''.join(first_half[::-1])
            else:
                full_string = ''.join(first_half) + ''.join(first_half[:-1][::-1])
            # Evaluate the string as a formula
            try:
                if eval(full_string) == N:
                    return full_string
            except:
                continue
    return -1

# Read input
N = int(input())
# Find the palindrome formula
result = find_palindrome_formula(N)
# Print the result
print(result)