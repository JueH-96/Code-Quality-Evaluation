def generate_palindrome(N):
    # For N=1, only possible answer is "="
    if N == 1:
        return "="
    
    # For N=2, only possible answer is "=="
    if N == 2:
        return "=="
    
    # For all other cases:
    # We'll place either one = or == in the middle/near-middle
    # and fill the rest with -
    
    # Create list of chars that we'll fill
    result = ['-'] * N
    
    # If N is odd
    if N % 2 == 1:
        # Place single = at center
        mid = N // 2
        result[mid] = '='
    else:
        # For even N, place == at middle two positions
        mid = N // 2
        result[mid-1:mid+1] = ['=', '=']
    
    return ''.join(result)

# Read input
N = int(input())

# Generate and print answer
print(generate_palindrome(N))