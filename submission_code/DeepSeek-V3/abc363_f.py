# YOUR CODE HERE
def find_palindrome_formula(N):
    # Since the string must be a palindrome and start with a digit, we can try to construct it
    # by considering the possible ways to form a palindrome that evaluates to N.
    # Given the constraints, we need an efficient way to generate such strings.
    
    # One approach is to consider the possible lengths of the string and try to construct
    # palindromes of those lengths that evaluate to N.
    
    # However, given the time constraints, we will limit the length to a manageable size.
    # For this problem, we will limit the length to 20 characters.
    
    max_length = 20
    
    # We will try to construct palindromes of length up to max_length
    for length in range(1, max_length + 1):
        # Generate all possible palindromes of the given length
        # Since it's a palindrome, we only need to generate the first half
        half_length = (length + 1) // 2
        # The first character must be a digit
        for first_char in '123456789':
            # Generate all possible combinations for the first half
            # We will use a recursive approach to generate all possible combinations
            # of digits and '*' for the first half
            def generate_combinations(current, index):
                if index == half_length:
                    # Construct the full palindrome
                    if length % 2 == 0:
                        full = current + current[::-1]
                    else:
                        full = current + current[:-1][::-1]
                    # Evaluate the expression
                    try:
                        value = eval(full)
                        if value == N:
                            return full
                    except:
                        pass
                    return None
                # Try to add a digit or '*'
                for c in '123456789*':
                    if c == '*' and index == 0:
                        continue  # first character must be a digit
                    result = generate_combinations(current + c, index + 1)
                    if result is not None:
                        return result
                return None
            result = generate_combinations(first_char, 1)
            if result is not None:
                return result
    return -1

# Read input
N = int(input())
# Find the palindrome formula
result = find_palindrome_formula(N)
# Print the result
print(result)